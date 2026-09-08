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
"""GFX950 attention config.

Supports both prefill (token_gen=False) and decode (token_gen=True).

Matches amd/mha.mojo config target:
  full_kv=True, depth_padded=False for both.
  Prefill:  double_buffer=True.
  Decode:   double_buffer=False, double_buffer_k_only when BN<=64,
            shared_kv only at depth>256 (SMEM budget).
"""

from std.math.uutils import umod, ufloordiv
from max.gpu import block_idx, lane_id
from std.utils import IndexList
from nn.attention.mha_utils import MHAConfig


# M (and N) of the wide decode MFMA, and the fold's `WM` on that arm: one 32-row
# tile per warp against the narrow arm's 16, which is how warps per CTA halve.
comptime _MHA_DECODE_WIDE_MMA_M = 32

# K of the wide decode MFMA, and so the fold's `BK` there: one MMA strip per
# SMEM block, which makes the two the same number by construction.
comptime _MHA_DECODE_WIDE_MMA_K = 64

# The narrow arm's `WM`, and the row quantum `_mha_decode_fold_ok` counts in.
comptime _MHA_DECODE_FOLD_WM = 16


@always_inline
def mha_decode_fold_wide_mma[
    dtype: DType, num_heads: Int, group: Int, q_seq_len: Int
]() -> Bool:
    """Whether the MHA decode token fold runs the 32x32x64 MFMA.

    MHA only: MLA decode folds too, but with its own block geometry.

    At `MMA_M == 32` two 16-f32 C-fragments join into the PV A-operand, so P
    stays in registers where the 16-row shape must round-trip it through LDS.
    `_mha_decode_fold_warp_m` reads this to pick `WM`, so the widths on the wide
    MMA and the widths given a 32-row M-tile are one set by construction.

    Rows must tile 32 — which also puts exactly one M-tile in a warp, all
    `PRegisterBuffer.mma_tile`'s gather handles — and leave at least THREE
    warps. Two suffices for the kernel but not for the split-K partition count,
    which callers derive from a 4-warp CTA (`hip_mha_decoding_num_partitions`):
    a 2-warp CTA handed that count measures +4% against the -22% of its own.
    `group == num_heads` selects the single-KV-head arm, the only one stacking
    `num_heads * q_seq_len` rows and setting `WN == BN`.

    Parameters:
        dtype: Element type shared by Q, K, and V.
        num_heads: Number of query heads, all owned by the fold's one KV head.
        group: Query heads per KV head.
        q_seq_len: Token SLOTS the M dimension is built from. Callers pass
            the PADDED tile width, not the tokens a sequence carries — the
            two differ exactly when `mha_decode_fold_tile_q_seq_len` pads.

    Returns:
        `True` when this fold width takes the 32x32x64 MFMA.
    """
    comptime rows = num_heads * q_seq_len
    return (
        dtype.is_float8()
        and q_seq_len > 1
        and group == num_heads
        and rows % _MHA_DECODE_WIDE_MMA_M == 0
        and rows >= 3 * _MHA_DECODE_WIDE_MMA_M
    )


@always_inline
def _mha_decode_fold_warp_m[
    dtype: DType, num_heads: Int, group: Int, S: Int
]() -> Int:
    """Return the warp M-tile height the single-KV-head fold arm launches with.

    One warp per 16-row M-tile makes warps-per-CTA equal `S`, and `WN == BN`
    gives every warp the whole key tile — so the CTA re-reads its KV tile out of
    LDS `S` times. 32 rows per warp halves that: -25 to -26% on MI355 at
    MiniMax-M3's fp8 decode shape, where LDS is the hottest unit.

    The 32 rows are ONE tile of the wide MFMA (see `mha_decode_fold_wide_mma`,
    which is also the gate), never two narrow ones. bf16 is excluded: it doubles
    every SMEM term, so halving the warps would only halve waves per CU.

    Parameters:
        dtype: Element type shared by Q, K, and V.
        num_heads: Number of query heads, all owned by this arm's one KV head.
        group: Query heads per KV head, `num_heads` on this arm.
        S: Token SLOTS the M dimension is built from, i.e. the PADDED tile
            width rather than the tokens a sequence carries.

    Returns:
        `WM` for the launch, a multiple of `_MHA_DECODE_FOLD_WM`.
    """
    return _MHA_DECODE_WIDE_MMA_M if mha_decode_fold_wide_mma[
        dtype, num_heads, group, S
    ]() else _MHA_DECODE_FOLD_WM


# Query rows the fold's M dimension may hold, one warp per M-tile: enough for a
# 16-query-head 1+7 verify. `_mha_decode_fold_ok` enforces it on the width a
# sequence carries, `mha_decoding` on the PADDED tile that actually launches.
comptime _MHA_DECODE_FOLD_MAX_ROWS = 8 * _MHA_DECODE_FOLD_WM

# Rows in the fold's widest tile, four warps of the wide MFMA. Sits exactly on
# `_MHA_DECODE_FOLD_MAX_ROWS`; lowering that without lowering this would trip
# the kernel's assert.
comptime _MHA_DECODE_WIDE_TILE_ROWS = 4 * _MHA_DECODE_WIDE_MMA_M

# Narrowest width worth padding, measured not derived: 48 rows (a 3-token fold
# at 16 heads) wins 12-13%, while the 32-row step below costs 0.104 ms against
# the full tile's 0.146. Bounds the SOURCE width, which need not tile 32 — only
# the padded tile it lands on does.
comptime _MHA_DECODE_PAD_MIN_ROWS = 48


@always_inline
def mha_decode_fold_tile_q_seq_len[
    dtype: DType, num_heads: Int, group: Int, q_seq_len: Int
]() -> Int:
    """Query token SLOTS the fold gives a width, padding narrow ones up.

    A width below the full tile runs dead rows on the four-warp 32x32x64 tile
    rather than taking a narrower geometry. Cost is set by the KV stream, not
    the row count, so dead rows are near-free (128 rows costs 0.148 ms where 96
    costs 0.205), and padding keeps the CTA at four warps — the one geometry
    term the host's split-K partition count assumes and cannot see.

    The result is the tile's HEIGHT only; the tokens a sequence carries stay
    with `mha_decoding`'s `q_seq_len`, which owns every stride. Host and kernel
    both call this rather than passing the two widths alongside each other.

    Parameters:
        dtype: Element type shared by Q, K, and V.
        num_heads: Number of query heads, all owned by the fold's one KV head.
        group: Query heads per KV head.
        q_seq_len: Query tokens the caller asked for.

    Returns:
        Token slots to build the tile from; `q_seq_len` when it is not padded.
    """
    comptime rows = num_heads * q_seq_len
    comptime if (
        dtype.is_float8()
        and q_seq_len > 1
        and group == num_heads
        and rows >= _MHA_DECODE_PAD_MIN_ROWS
        and rows < _MHA_DECODE_WIDE_TILE_ROWS
        and _MHA_DECODE_WIDE_TILE_ROWS % num_heads == 0
    ):
        return _MHA_DECODE_WIDE_TILE_ROWS // num_heads
    return q_seq_len


@always_inline
def decode_mma_shape[
    dtype: DType,
    depth: Int,
    num_heads: Int,
    mla_mode: Bool = False,
    fold_wide_mma: Bool = False,
]() -> IndexList[3]:
    """Return the MFMA shape the gfx950 decode kernels use for this shape.

    Split out of `AMDStructuredConfig.get_mma_shape` so host-side dispatch can
    ask for the shape without building a full config, and so there is one
    definition of the rule rather than two that can drift.

    Parameters:
        dtype: Element type shared by Q, K, and V.
        depth: Attention head depth.
        num_heads: Number of query heads.
        mla_mode: Whether multi-latent attention tiling is active.
        fold_wide_mma: Whether the MHA token fold's wide arm applies, from
            `mha_decode_fold_wide_mma`. Defaults to False, which asks for the
            unfolded shape — what the host fold gate tests before it knows the
            width.

    Returns:
        The `(M, N, K)` MFMA shape.
    """
    # The MHA token fold's wide arm overrides the depth rule below: it keeps P
    # in registers, which needs a 32-row MFMA.
    comptime if fold_wide_mma:
        return IndexList[3](
            _MHA_DECODE_WIDE_MMA_M,
            _MHA_DECODE_WIDE_MMA_M,
            _MHA_DECODE_WIDE_MMA_K,
        )
    comptime if dtype.is_float8():
        # MLA decode with `num_heads <= 16` packs at most one MFMA row group,
        # so `16x16x128` with `BM=WM=16` puts one warp on one full tile with
        # no wasted M lanes (Kimi-K2.5 per-GPU under TP=4 lands at exactly 16
        # query heads). Otherwise prefer `16x16x128` when `depth % 128 == 0`
        # and fall back to `32x32x64` for full M-dim utilization.
        comptime if (mla_mode and num_heads <= 16) or depth % 128 == 0:
            return IndexList[3](16, 16, 128)
        return IndexList[3](32, 32, 64)
    # BF16 decode: 16x16x32 regardless of depth.
    return IndexList[3](16, 16, 32)


@fieldwise_init
struct AMDStructuredConfig[
    config: MHAConfig,
    group: Int,
    token_gen: Bool = False,
    mla_mode: Bool = False,
    q_seq_len: Int = 1,
](ImplicitlyCopyable):
    """Holds the tile layout and indexing helpers for GFX950 structured attention.

    Wraps an `MHAConfig` with group-query and MLA metadata, deriving the
    comptime shared-memory decisions (`shared_kv`, `full_kv`,
    `depth_padded`, `double_buffer`) and exposing query/key-value head and
    tile index helpers used by the attention kernels.

    Parameters:
        config: The base multi-head attention configuration.
        group: Number of query heads sharing one key-value head.
        token_gen: Selects decode (`True`) versus prefill (`False`) tiling.
        mla_mode: Enables multi-latent attention tiling when `True`.
        q_seq_len: Token slots the decode fold stacks into M — the tile's
            height, padded above a sequence's real token count on narrow
            widths; selects the MFMA shape for the MHA fold's wide arm.
    """

    comptime shared_kv = Self.token_gen and Self.config.depth > 256
    comptime full_kv = True
    comptime depth_padded = False
    comptime double_buffer = not Self.token_gen
    comptime double_buffer_k_only = (
        Self.token_gen and Self.config.block_n() <= 64
    )

    @staticmethod
    @always_inline
    def heads_per_tile() -> Int:
        # MLA: tile spans `BM` heads of the single latent kv (block_idx.y is
        # the tile idx). MHA: tile spans `group` heads of one kv head
        # (block_idx.y is the kv-head idx).
        return Self.config.block_m() if Self.mla_mode else Self.group

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        comptime if Self.token_gen:
            comptime mma_shape = Self.get_mma_shape()
            var lane_in_row = umod(lane_id(), mma_shape[0])
            return Int(block_idx.y) * Self.heads_per_tile() + lane_in_row
        else:
            return block_idx.x

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        comptime if Self.token_gen:
            # MLA decode tiles queries across block_idx.y; MHA decode keeps
            # all queries of the kv-head in block 0.
            return Int(block_idx.y) if Self.mla_mode else 0
        else:
            return Int(block_idx.y)

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        comptime if Self.token_gen:
            # MLA decode: single latent kv head at index 0.
            return 0 if Self.mla_mode else Int(block_idx.y)
        else:
            return ufloordiv(Self.q_head_idx(), Self.group)

    @staticmethod
    @always_inline
    def get_mma_shape() -> IndexList[3]:
        comptime if Self.token_gen:
            return decode_mma_shape[
                Self.config.dtype,
                Self.config.depth,
                Self.config.num_heads,
                Self.mla_mode,
                fold_wide_mma=(
                    not Self.mla_mode
                    and mha_decode_fold_wide_mma[
                        Self.config.dtype,
                        Self.config.num_heads,
                        Self.group,
                        Self.q_seq_len,
                    ]()
                ),
            ]()
        # FP8 prefill: 32x32x64.  BF16 prefill: 32x32x16.
        comptime if Self.config.dtype.is_float8():
            return IndexList[3](32, 32, 64)
        return IndexList[3](32, 32, 16)

    @staticmethod
    @always_inline
    def get_q_offset[q_depth: Int]() -> UInt32:
        comptime if Self.token_gen and Self.mla_mode:
            # MLA decode: `BM` queries per tile along block_idx.y.
            return UInt32(q_depth * Self.q_tile_idx() * Self.config.block_m())
        return UInt32(
            q_depth
            * (
                (
                    Self.kv_head_idx()
                    * Self.group if Self.token_gen else Self.q_head_idx()
                )
                + Self.config.num_heads
                * Self.q_tile_idx()
                * Self.config.block_m()
            )
        )

    @staticmethod
    @always_inline
    def get_output_offset[output_depth: Int]() -> UInt32:
        return Self.get_q_offset[output_depth]()
