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
"""Named TileOp struct for QK score masking on gfx950.

Uses TileLayout / `Coord` Layout Algebra (not hand-rolled integer formulas)
to map lanes and registers into the MMA fragment space:

- `WarpLayoutT.idx2crd(lane)` decomposes the lane into (lane_row, lane_col).
- `FragmentLayoutT(Idx[j])` maps register `j` to its column offset within
  the MMA fragment.
- `FragmentLayoutT.static_product` / `static_shape[i]` expose the fragment
  size and the per-lane column-group stride.

Fragment layout differs by MMA size:

- 16x16 MMA: 4 regs/lane, flat `(1, 4):(1, 1)`.
- 32x32 MMA: 16 regs/lane organized as 4 groups of 4 cols with stride 8
  between groups: nested `((1,(4,4)):(1,(1,8)))` (fp8 MFMA pattern).
"""

from std.math.constants import log2e
from std.math.uutils import umod
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from max.gpu import block_idx, lane_id
from std.utils import IndexList

from layout import TileTensor
from layout.tile_layout import Layout, col_major
from layout.coord import Coord, ComptimeInt, Idx
from nn.attention.mha_mask import CausalMask, MHAMask
from nn.attention.mha_utils import _kernel_mask


struct MaskTileOp[
    accum_type: DType,
    token_gen: Bool,
    mma_shape: IndexList[3],
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    # MMA M extent (32 or 16 on gfx950) — selects lane + fragment geometry.
    mma_m: Int,
    use_exp2: Bool = False,
    # Token-fold geometry. `num_heads_per_token` = H maps a folded row to its
    # token/head (token = row // H, head = row % H); `valid_rows` = M = H*S live
    # rows. Defaults (group, group) reproduce single-token decode (S=1).
    num_heads_per_token: Int = group,
    valid_rows: Int = group,
    # Query heads one grid_y step advances past, for the folded row -> head map:
    # `group` for MHA (a CTA per KV head), 0 for MLA (grid_y is a query tile).
    head_base_stride: Int = 0,
    # The fold applies only to AMD decode (MLA always, MHA at S>1). False on the
    # shared prefill / single-token-decode callers, where it makes the fold
    # arithmetic comptime-dead (byte-identical legacy mask).
    fold_mode: Bool = False,
    # Warps splitting the MMA M dimension (= BM // WM). At 1 (legacy + all shared
    # callers) the absolute query row equals intra-warp `lane_row`, so the cheap
    # top-level `if lane_row >= valid_rows: return` is exact. At >1 (warp-local)
    # warp 1's `lane_row` is 0..15 but its absolute rows are 16..31, so the
    # dead-row guard must instead test the absolute row per-fragment (see below).
    num_warps_m: Int = 1,
]:
    """Apply an `MHAMask` to the per-lane QK score registers in place.

    Parameters:
        accum_type: The element type of the QK score registers and `scale`.
        token_gen: Whether the kernel runs in token-generation (decode) mode
            versus prefill.
        mma_shape: The MMA tile shape `(M, N, K)` used to compute per-fragment
            row and column offsets.
        num_m_mmas: Number of MMAs along the M (query-row) dimension of the
            score tile.
        num_n_mmas: Number of MMAs along the N (key-column) dimension of the
            score tile.
        mask_t: The `MHAMask` type to apply to the score registers.
        group: GQA group size; the number of query heads sharing one KV head.
        mma_m: MMA M extent (32 or 16 on gfx950) that selects the lane and
            fragment geometry.
        use_exp2: Whether the consumer softmax uses base-2 exponentiation
            (defaults to False).
        num_heads_per_token: Token-fold geometry; H heads per token mapping a
            folded row to its token and head, where token = row // H and
            head = row % H (defaults to `group`).
        valid_rows: Query rows the TILE spans, M = H * S across all heads and
            sequence positions (defaults to `group`). On a padded fold tile
            this exceeds the live rows, which are the runtime `H * seq_len`.
        head_base_stride: Query heads spanned by one `grid_y` step, used to base
            a folded row's head index (defaults to 0).
        fold_mode: Whether the token fold applies, for AMD decode only; when
            False the fold arithmetic is comptime-dead (defaults to False).
        num_warps_m: Number of warps splitting the MMA M dimension (= BM //
            WM), which controls the dead-row guard strategy (defaults to 1).
    """

    # Warp lane layout (col-major (warp_rows, warp_cols)) as a proper
    # TileLayout. idx2crd decomposes a lane into (lane_row, lane_col).
    comptime _warp_rows = 32 if Self.mma_m == 32 else 16
    comptime _warp_cols = 2 if Self.mma_m == 32 else 4
    comptime WarpLayoutT = type_of(
        col_major[Self._warp_rows, Self._warp_cols]()
    )

    # Per-lane MMA fragment layout, unified across 16x16 and 32x32 MMAs.
    #
    # 32x32 fp8 MFMA produces 16 regs/lane in 4 groups of 4 cols with stride
    # 8 between groups; 16x16 produces 4 contiguous regs/lane. Both are
    # expressed as `((1, (4, _outer_groups)) : (1, (1, 8)))`:
    # - 32x32: _outer_groups = 4  → shape (1, 16), cols (j%4) + (j//4)*8.
    # - 16x16: _outer_groups = 1  → shape (1, 4),  cols = j (outer dim is a
    #   single group, so the stride-8 term always contributes 0).
    #
    # Using one shape means the struct field has one concrete type — this
    # sidesteps the "ternary between different Layout specializations fails
    # to unify" issue (feedback_mojo_conditional_field_type).
    comptime _outer_groups = 4 if Self.mma_m == 32 else 1
    comptime FragmentLayoutT = Layout[
        Coord[
            ComptimeInt[1],
            Coord[ComptimeInt[4], ComptimeInt[Self._outer_groups]],
        ].element_types,
        Coord[
            ComptimeInt[1], Coord[ComptimeInt[1], ComptimeInt[8]]
        ].element_types,
    ]

    comptime output_frag_size = Self.FragmentLayoutT.static_product

    # Stride (in fragment cols) between adjacent lane-column groups: 4.
    # This is shape[1][0] of the nested fragment layout for both MMA sizes.
    comptime _lane_col_stride = 4

    @staticmethod
    @always_inline
    def apply(
        masked: Bool,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        start_pos: UInt32,
        seq_len: UInt32,
        num_keys: UInt32,
        mask_block_row: UInt32,
        mask_warp_row: UInt32,
        mask_warp_col: UInt32,
        scale: Scalar[Self.accum_type],
        mask: Self.mask_t,
        p_reg_tile: TileTensor[mut=True, Self.accum_type, ...],
        not_last_iter: Bool,
        cache_start_pos: UInt32 = 0,
    ):
        comptime output_frag_size = Self.output_frag_size
        comptime assert p_reg_tile.flat_rank == 2
        var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()
        comptime assert p_reg_vectorized.flat_rank == 2

        # Decompose lane via the warp layout: WarpLayout is col-major
        # (warp_rows, warp_cols), so idx2crd(lane) = (lane_row, lane_col).
        var lane_coord = Self.WarpLayoutT().idx2crd(Int(lane_id()))
        var lane_row = Int(lane_coord[0].value())
        var lane_col = Int(lane_coord[1].value())
        var lane_col_off = lane_col * Self._lane_col_stride

        # Token fold: S = valid_rows / H (= 1 off the fold path, byte-identical).
        comptime fold_seq_len = (
            Self.valid_rows // Self.num_heads_per_token
        ) if Self.fold_mode else 1

        # Variable per-sequence query length: for S>1 the comptime `valid_rows =
        # H * q_seq_len` is the ceiling, but a short sequence has fewer live rows.
        # `seq_len` is its runtime query length, so the live-row count is
        # `H * seq_len`. The `fold_seq_len > 1` dead-row guards and causal
        # `score_row` use these runtime values; `fold_seq_len == 1` keeps the
        # comptime path. The two counts coincide only on a uniform batch whose
        # tile is UNPADDED, so never substitute one for the other.
        var valid_rows_rt = UInt32(Self.num_heads_per_token) * seq_len

        # Decode: skip dead query rows. At num_warps_m == 1 the absolute row ==
        # `lane_row`, so the top-level early-out is exact (bound generalized from
        # `group` to `valid_rows`, or the runtime `valid_rows_rt` on the S>1
        # fold). At num_warps_m > 1 the intra-warp test can't filter warp 1's pad
        # rows, so the guard moves into the loop as a per-fragment absolute-row
        # check (below; padding keeps finite/zero scores, no top-level return).
        comptime if Self.token_gen and Self.num_warps_m == 1:
            comptime if fold_seq_len > 1:
                if UInt32(lane_row) >= valid_rows_rt:
                    return
            else:
                if lane_row >= Self.valid_rows:
                    return

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma

                var mask_frag_row = mask_warp_row + UInt32(
                    m_mma * Self.mma_shape[0]
                )
                var mask_frag_col = (
                    mask_warp_col
                    + UInt32(n_mma * Self.mma_shape[1])
                    + (kv_tile_start_row if Self.token_gen else 0)
                )
                mask_frag_row += UInt32(lane_row)
                mask_frag_col += UInt32(lane_col_off)
                # Dead-row guard for warp-local: a warp's 16 lanes may include
                # pad rows >= valid_rows the top-level return can't filter (the
                # warp also owns live rows). Guard the per-fragment store on the
                # absolute row; dead rows keep their raw (~0, finite) QK score and
                # are dropped by the valid_rows SRD clamp at store time. At
                # num_warps_m==1 `_row_live` is comptime True (unconditional
                # store, byte-identical).
                comptime _row_live_static = (
                    not Self.token_gen
                ) or Self.num_warps_m == 1
                # S>1 fold: bound is the runtime `valid_rows_rt` (drops
                # short-sequence pad rows); `fold_seq_len == 1` keeps the
                # comptime `valid_rows`. Neither is read when num_warps_m==1.
                var _row_live: Bool
                comptime if fold_seq_len > 1:
                    _row_live = _row_live_static or (
                        mask_frag_row < valid_rows_rt
                    )
                else:
                    _row_live = _row_live_static or (
                        Int(mask_frag_row) < Self.valid_rows
                    )
                # Per-token causal position (heads-inner fold): token = row // H,
                # score_row = num_keys - S + token = mha_gpu_naive's
                # `y + cur_cache_len - cur_query_len`. S==1 keeps the literal
                # `num_keys - 1` (byte-identical); prefill uses the Q tile pos.
                var score_row: UInt32
                comptime if Self.token_gen:
                    comptime if fold_seq_len == 1:
                        score_row = num_keys - 1
                    else:
                        var _token = mask_frag_row // UInt32(
                            Self.num_heads_per_token
                        )
                        # Runtime `seq_len`, not comptime `fold_seq_len`:
                        # `num_keys - seq_len + token == cache_len + token`.
                        # Load-bearing — a padded tile carries more slots than
                        # tokens, so the constant would walk every causal
                        # position back by the pad and over-mask newest keys.
                        score_row = num_keys - seq_len + _token
                else:
                    score_row = mask_block_row + mask_frag_row
                var score_col = mask_frag_col
                var score_col_with_cache_start_pos = score_col + cache_start_pos
                var score_row_with_start_pos = score_row + start_pos
                comptime is_causal_mask = Self.mask_t == CausalMask

                # Load the whole MMA fragment once, mutate lanes on a local
                # SIMD (which `SIMD.__setitem__` supports), and store back
                # once at the end. Chained `vec[i, 0][j] = ...` on a raw
                # TileTensor vectorize would write into a by-value copy and
                # silently drop the update.
                var frag = p_reg_vectorized[mma_id, 0]

                if masked:
                    comptime if is_causal_mask and not Self.token_gen:
                        var x_0 = Int32(
                            score_row_with_start_pos
                            - score_col_with_cache_start_pos
                        )
                        comptime for j in range(0, output_frag_size, 2):
                            # Fragment column offset for register j/j+1 from
                            # the fragment Layout itself (Layout.__call__).
                            comptime y_0 = (
                                Int32(Self.FragmentLayoutT()(Idx[j])) - 1
                            )
                            var val_0 = frag[j]
                            comptime y_1 = (
                                Int32(Self.FragmentLayoutT()(Idx[j + 1])) - 1
                            )
                            var val_1 = frag[j + 1]
                            comptime asm = """
                                    v_mov_b32 $4, 0xc61c4000
                                    v_cmp_lt_i32_e64 $2, $6, $5
                                    v_cmp_lt_i32_e64 $3, $9, $8
                                    v_cndmask_b32_e64 $0, $4, $7, $2
                                    v_cndmask_b32_e64 $1, $4, $10, $3
                                    """
                            var ret = inlined_assembly[
                                asm,
                                _RegisterPackType[
                                    Scalar[Self.accum_type],
                                    Scalar[Self.accum_type],
                                    Int64,
                                    Int64,
                                    Scalar[Self.accum_type],
                                ],
                                constraints="=v,=v,=&s,=&s,=&v,v,n,v,v,n,v,~{vcc}",
                            ](x_0, y_0, val_0, x_0, y_1, val_1)
                            frag[j] = ret[0]
                            frag[j + 1] = ret[1]

                    else:
                        # Prefill: q_head_idx = block_idx.x. Decode: block_idx.y
                        # * group + lane_group; fold (S>1): the row's head based
                        # at its KV head's first. S==1 is split out
                        # (byte-identical). Only head-keyed masks consume this;
                        # Causal/Null ignore it.
                        var q_head_idx = Int(block_idx.x)
                        comptime if Self.token_gen:
                            comptime if fold_seq_len == 1:
                                q_head_idx = block_idx.y * Self.group + umod(
                                    lane_id(), Self.mma_shape[0]
                                )
                            else:
                                # Mirrors `Attention`'s `r_abs` stat key. At
                                # `group == 1` H is 1, so `row % H` alone
                                # collapses every row to head 0; the base term
                                # supplies the CTA's KV head.
                                var head_in_token = (
                                    Int(mask_frag_row)
                                    % Self.num_heads_per_token
                                )
                                q_head_idx = (
                                    Int(block_idx.y) * Self.head_base_stride
                                    + head_in_token
                                )
                        comptime for j in range(0, output_frag_size, 1):
                            comptime fragment_col = Int(
                                Self.FragmentLayoutT()(Idx[j])
                            )
                            frag[j] = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    block_idx.z,
                                    q_head_idx,
                                    Int(score_row_with_start_pos),
                                    Int(
                                        score_col_with_cache_start_pos
                                        + UInt32(fragment_col)
                                    ),
                                ),
                                frag[j],
                            )

                comptime if Self.mask_t.apply_log2e_after_mask:
                    frag = frag * log2e

                if (
                    not not_last_iter or Self.token_gen
                ) and Self.mask_t.mask_out_of_bound:
                    var bound_y = (
                        kv_tile_start_row
                        + kv_tile_num_rows if Self.token_gen else num_keys
                    )

                    comptime for j in range(output_frag_size):
                        comptime fragment_col = Int(
                            Self.FragmentLayoutT()(Idx[j])
                        )

                        var bound_x = num_keys if Self.token_gen else seq_len

                        frag[j] = _kernel_mask(
                            IndexList[2, element_type=.uint32](
                                Int(score_row),
                                Int(score_col + UInt32(fragment_col)),
                            ),
                            IndexList[2, element_type=.uint32](
                                Int(bound_x), Int(bound_y)
                            ),
                            frag[j],
                        )

                # Skip the store for warp-local pad rows (dropped at store time
                # anyway). `_row_live` is comptime True at num_warps_m==1, so the
                # store is unconditional (byte-identical).
                if _row_live:
                    p_reg_vectorized[mma_id, 0] = frag
