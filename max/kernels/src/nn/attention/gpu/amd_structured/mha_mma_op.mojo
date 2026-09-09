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
"""MHA MMA operator: shape constants, SMEM→register loaders, and
MFMA dispatch used by `MhaPrefillV2`.

Supports two MFMA flavors on gfx950, comptime-selected on `T`:

- BF16 (`v_mfma_f32_32x32x16_bf16`): `MMA_K=16`, 8 BF16 elts/lane/base.
  K side uses the two-XOR worker-index swizzle and `ds_read_b128`;
  V side uses the identity swizzle and the `ds_read_tr16_b64_warp`
  transpose-load.
- FP8 e4m3 (`v_mfma_scale_f32_32x32x64_f8f6f4`): `MMA_K=64`, 32 FP8
  elts/lane/base. K loader reuses the BF16 byte-level two-XOR
  swizzle, the byte-positional swizzle works for both sizes
  because the sub-block is 64 B-wide either way. FP8 V loader uses
  `ds_read_tr8_b64` paired-lane transpose-loads, mirroring
  `TiledMmaLoader.load_v_fp8_strip`. `mma_QK` / `mma_PV` are
  dtype-generic; `gpu_mma` dispatches on SIMD operand sizes.
"""

from max.gpu import lane_id
from max.gpu.compute.mma import mma as gpu_mma
from max.gpu.intrinsics import ds_read_tr8_b64
from std.math import exp2 as math_exp2
from std.memory import AddressSpace
from std.sys import size_of
from std.utils import IndexList

from layout import Coord, TensorLayout
from layout.coord import crd2idx
from layout.tile_layout import row_major

from structured_kernels.amd_tile_io import (
    RegTile,
    SMemTile,
    _load_from_lds,
    ds_read_tr16_b64_warp,
)


# `v_mfma_f32_32x32x16_bf16` on gfx950 distributes its 32×32 FP32 output
# across wave64 such that lane `L` owns 16 elements at columns `L & 31`
# and rows `ACC_ROW_OFFSETS_32x32[p] + ((L >> 5) << 2)` for `p` in
# `[0, 16)`: four stripes of four rows, spaced 8 rows apart, with the
# high half-warp shifted by 4. Hardware-determined; consumers needing to
# map per-lane fragment indices to matrix rows should read this table.
comptime ACC_ROW_OFFSETS_32x32 = SIMD[.int32, 16](
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
)


struct MhaConfigV2(ImplicitlyCopyable, Movable):
    """Shape configuration for `MhaPrefillV2`.

    Single source of truth for the shape parameters that drive
    register-tile layouts, SMEM sub-block geometry, grid dimensions,
    and IGLP scheduling. Lives in `mha_mma_op` so both `MhaMmaOp`
    and `MhaPrefillV2` can take it as a parameter without a circular
    import.
    """

    var q_block_size: Int
    """Q rows per warp."""

    var kv_block: Int
    """K/V rows per tile (64 at d=128)."""

    var depth: Int
    """Head depth."""

    var num_heads: Int
    """Q num_heads."""

    var num_kv_heads: Int
    """K/V num_heads. `1` (full GQA) or equal to `num_heads` (MHA);
    other ratios need a stride-aware DMA loader (TODO)."""

    var num_warps: Int
    """Warps per block."""

    var rescale_threshold: Float32
    """Lazy-rescale threshold in log2 units of the running max. Above
    this, `o_reg` / `norm_vec` are rescaled by
    `exp2(max_prev - max_new)`; below this the rescale is deferred:
    the residual contribution from `att_block` is bounded by
    `exp2(-rescale_threshold)` of the new max."""

    var dtype: DType
    """Element dtype of the Q / K / V input tiles. `DType.bfloat16` for
    the production BF16 prefill path; `DType.float8_e4m3fn` for the
    FP8 prefill path. Comptime-dispatched throughout
    `MhaMmaOp` (MFMA shape, swizzle, SMEM sub-block dims) and
    `MhaPrefillV2` (register tiles, SMEM slots, cooperative loaders)."""

    var output_dtype: DType
    """Element dtype of the output tile `o`. FP32 by default; BF16 for
    production inference where the dispatcher holds a BF16 output buffer.
    The cast from the FP32 accumulator happens per-lane inside
    `_store_o_to_gmem`."""

    var fp8_mma_k_128: Bool
    """When True and `dtype` is an FP8 type, select the
    `v_mfma_scale_f32_16x16x128_f8f6f4` MFMA shape (MMA_M=MMA_N=16,
    MMA_K=128) instead of the default `v_mfma_scale_f32_32x32x64_f8f6f4`
    (MMA_M=MMA_N=32, MMA_K=64).

    The 16×16×128 shape issues every 16 cycles vs 32 cycles for
    32×32×64; mirrors the FP8 ping-pong / 4-wave matmul choice
    (see `amd_ping_pong_matmul.mojo:716-737`). BF16 is unaffected;
    it always uses 32×32×16. Defaults to False (today's 32×32×64
    path)."""

    def __init__(
        out self,
        *,
        q_block_size: Int,
        kv_block: Int,
        depth: Int,
        num_heads: Int,
        num_kv_heads: Int,
        num_warps: Int = 8,
        rescale_threshold: Float32 = 8.0,
        dtype: DType = .bfloat16,
        output_dtype: DType = .float32,
        fp8_mma_k_128: Bool = False,
    ):
        self.q_block_size = q_block_size
        self.kv_block = kv_block
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_warps = num_warps
        self.rescale_threshold = rescale_threshold
        self.dtype = dtype
        self.output_dtype = output_dtype
        self.fp8_mma_k_128 = fp8_mma_k_128


struct MlaConfigV2(ImplicitlyCopyable, Movable):
    """Shape configuration for `MlaPrefillV2Core`. Companion to
    `MhaConfigV2`.

    DeepSeek-V3-style MLA: Q is concatenated `q_nope || q_rope` at
    `d_qk = d_nope + d_rope`. K is stored in a latent cache at
    `cache_depth` columns wide, with `k_nope` at `[:, :d_nope]` and
    `k_rope` at `[:, rope_cache_offset:rope_cache_offset + d_rope]`
    (the gap between the two segments is padded / reserved but counted
    in the cache stride). V is `v_nope` only, so V and O stay at
    `d_pv = depth`, identical to the MHA path; DeepSeek-V3 MLA does
    not RoPE V.

    The MFMA-shape / SMEM-sub-block / K-loader / V-loader / PV-path
    machinery is shared with `MhaPrefillV2` via
    `MhaMmaOp[T, config.mha()]`. `mha()` derives an `MhaConfigV2` from
    `Self` for that sharing. MLA's divergence is the Q load at
    `d_qk`, the two K segments, and the cluster schedule that
    interleaves `k_nope` / `k_rope` DMAs with V.

    The latent-cache layout (576-wide with `k_rope` at offset 512) is
    fixed by DeepSeek-V3; matches the existing BF16 MLA path in
    `mla_prefill.mojo` (`cache_depth = 576`, `head_dim_offset =
    cache_depth - rope_depth = 512`).
    """

    # Fields mirrored from MhaConfigV2 (`mha()` is a straightforward
    # field copy — keep declaration order in sync).
    var q_block_size: Int
    """Q rows per warp."""

    var kv_block: Int
    """K/V rows per tile (64 at `d_pv=128`)."""

    var depth: Int
    """V / O head depth (`d_pv = d_nope`). For DeepSeek-V3 MLA: 128."""

    var num_heads: Int
    """Q num_heads."""

    var num_kv_heads: Int
    """K/V num_heads. `1` (full GQA) or equal to `num_heads` (MHA);
    other ratios need a stride-aware DMA loader (TODO)."""

    var num_warps: Int
    """Warps per block."""

    var rescale_threshold: Float32
    """Lazy-rescale threshold in log2 units of the running max
    (identical semantics to `MhaConfigV2.rescale_threshold`)."""

    var dtype: DType
    """Element dtype of Q / K (both `q_nope ∥ q_rope` and `k_nope ∥
    k_rope`) / V input tiles. `DType.bfloat16` for parity with the
    existing BF16 MLA prefill; `DType.float8_e4m3fn` for the
    FP8 MLA prefill path."""

    var output_dtype: DType
    """Element dtype of the output tile `o`. FP32 by default; BF16
    for inference dispatchers holding a BF16 output buffer. The cast
    from the FP32 accumulator happens per-lane inside the output
    store."""

    var fp8_mma_k_128: Bool
    """Mirror of `MhaConfigV2.fp8_mma_k_128`. Architecturally blocked
    for this attention path by the QK-output / PV-B-input lane geometry
    mismatch, kept for symmetry so MLA inherits the same comptime hook
    if a cross-lane shuffle becomes available."""

    # MLA-specific fields.
    var d_qk: Int
    """Q / K depth (`d_nope + d_rope`). For DeepSeek-V3 MLA: 192."""

    var d_rope: Int
    """RoPE-applied segment depth on Q and K. For DeepSeek-V3 MLA: 64."""

    var cache_depth: Int
    """Latent K cache row width. For DeepSeek-V3 MLA: 576. The gap
    between `d_nope` (128) and `rope_cache_offset` (512) is reserved
    / unused but present in the cache stride. Must match the
    production latent cache layout; see `mla_prefill.mojo:54`."""

    var rope_cache_offset: Int
    """Column offset of `k_rope` within the latent cache row. For
    DeepSeek-V3 MLA: 512 (layout: `k_nope` at `[:, :128]`, gap,
    `k_rope` at `[:, 512:576]`)."""

    def __init__(
        out self,
        *,
        q_block_size: Int,
        kv_block: Int,
        depth: Int,
        num_heads: Int,
        num_kv_heads: Int,
        d_qk: Int,
        d_rope: Int,
        cache_depth: Int,
        rope_cache_offset: Int,
        num_warps: Int = 8,
        rescale_threshold: Float32 = 8.0,
        dtype: DType = .bfloat16,
        output_dtype: DType = .float32,
        fp8_mma_k_128: Bool = False,
    ):
        self.q_block_size = q_block_size
        self.kv_block = kv_block
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_warps = num_warps
        self.rescale_threshold = rescale_threshold
        self.dtype = dtype
        self.output_dtype = output_dtype
        self.fp8_mma_k_128 = fp8_mma_k_128
        self.d_qk = d_qk
        self.d_rope = d_rope
        self.cache_depth = cache_depth
        self.rope_cache_offset = rope_cache_offset

    @always_inline
    def d_nope(self) -> Int:
        """Returns the non-RoPE segment depth (= `depth` = `d_pv`).

        For DeepSeek-V3 MLA `d_nope == depth == 128`. Exposed as an
        accessor so `MlaPrefillV2Core` body code can reference the
        nope-segment depth by its semantic name without committing to
        an additional field.
        """
        return self.depth

    @always_inline
    def mha(self) -> MhaConfigV2:
        """Returns an `MhaConfigV2` derived from `Self` for sharing
        the `MhaMmaOp[T, ...]` machinery.

        The MFMA shape, SMEM sub-block geometry, K loader, V loader,
        and PV path all live on `MhaMmaOp`; MLA's divergence is
        purely in `MlaPrefillV2Core` (Q load at `d_qk`, K_rope DMA,
        two-segment QK). The derived `MhaConfigV2` carries `depth =
        d_pv = d_nope`; the MLA-specific `d_qk` / `d_rope` /
        `cache_depth` / `rope_cache_offset` fields stay on
        `MlaConfigV2` only.
        """
        return MhaConfigV2(
            q_block_size=self.q_block_size,
            kv_block=self.kv_block,
            depth=self.depth,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            num_warps=self.num_warps,
            rescale_threshold=self.rescale_threshold,
            dtype=self.dtype,
            output_dtype=self.output_dtype,
            fp8_mma_k_128=self.fp8_mma_k_128,
        )


struct MhaMmaOp[T: DType, config: MhaConfigV2]:
    """Namespace-style struct holding the shape constants, register-tile
    layouts, and SMEM→register loaders for `MhaPrefillV2`. All call
    sites go through static methods on this struct.

    Supports two MFMA flavors on gfx950, comptime-selected on `T`:

    - BF16 → `v_mfma_f32_32x32x16_bf16`, `MMA_K=16`, 8 BF16 elts/lane.
    - FP8 e4m3 → `v_mfma_scale_f32_32x32x64_f8f6f4`, `MMA_K=64`, 32 FP8
      elts/lane. MMA dispatch is automatic via `gpu_mma` SIMD-size
      overload resolution; only the per-lane fragment size differs.

    `load_K` / `load_V` both comptime-branch on `T.is_float8()`:
    - BF16 path: byte-identical to the original reference kernel: K
      via two-XOR swizzle + `ds_read_b128`, V via
      `ds_read_tr16_b64_warp`.
    - FP8 path: K reuses the same byte-level two-XOR swizzle (the
      sub-block is byte-equivalent at 32 rows × 64 B), V uses
      `ds_read_tr8_b64` paired-lane transpose-loads matching
      `TiledMmaLoader.load_v_fp8_strip`.

    Parameters:
        T: Element data type (BF16 or FP8 e4m3).
        config: Shape configuration.
    """

    # Config aliases — body code references `Self.Q_BLOCK_SIZE` etc.
    # for readability; the values come from `config`.
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth

    # Whether FP8 path uses the alternative 16×16×128 MFMA shape.
    # When False (default), FP8 uses today's 32×32×64. When True, FP8
    # opens to MMA_M=MMA_N=16, MMA_K=128 (mirrors ping-pong matmul).
    # BF16 always uses 32×32×16 — this flag has no effect there.
    comptime FP8_MMA_K_128 = Self.config.fp8_mma_k_128

    # MFMA shape — comptime-selected on dtype and the FP8 MMA shape
    # flag:
    # - BF16: always 32×32×16 (`v_mfma_f32_32x32x16_bf16`).
    # - FP8 default (32×32×64): `v_mfma_scale_f32_32x32x64_f8f6f4`.
    # - FP8 alt (16×16×128): `v_mfma_scale_f32_16x16x128_f8f6f4`
    #   (when `config.fp8_mma_k_128 == True`).
    comptime MMA_M = 16 if Self.T.is_float8() and Self.FP8_MMA_K_128 else 32
    comptime MMA_N = 16 if Self.T.is_float8() and Self.FP8_MMA_K_128 else 32
    comptime MMA_K = (
        128 if Self.T.is_float8()
        and Self.FP8_MMA_K_128 else 64 if Self.T.is_float8() else 16
    )

    # Shape constants. Sub-block dims are the SMEM tile geometry that
    # matches the SMEM swizzle; FRAG_ELTS / ROWL_* are the per-lane
    # decomposition of the MFMA input fragments.

    comptime K_SUB_ROWS = 32
    """K SMEM sub-block rows. Same for BF16 and FP8 (32×32×64 path);
    the FP8 16×16×128 path keeps the same parent SMEM geometry so
    the cooperative DMA producer (`SubTileLoaderLDS`) doesn't change;
    the difference is purely on the consumer-side lane partition
    that `load_K` performs.

    BF16: 32×32 BF16 elts = 32×64 B. FP8 (any shape): 32×64 FP8 elts
    = 32×64 B (byte-equivalent so the byte-positional swizzle reuses
    across dtypes)."""

    comptime K_SUB_COLS = 64 if Self.T.is_float8() else 32
    """K SMEM sub-block cols. 32 BF16 elts = 64 B/row; 64 FP8 elts =
    64 B/row. Byte-equivalent, same SMEM geometry for both FP8 MMA
    shapes (32×32×64 and 16×16×128) because the parent allocation in
    `mha_prefill_v2.mojo` is unchanged; only the consumer-side lane
    partition differs.

    For FP8 32×32×64: each sub-block holds exactly one 32×64 base
    tile.

    For FP8 16×16×128: each sub-block holds two 16-row sub-strips
    (top + bottom 16 rows), each spanning 64 B. A single 16×128
    MFMA base tile needs 2 of these K-direction sub-blocks (to cover
    128 K elements at 64 cols/sub-block). The consumer-side lane
    partition is 16-lanes-per-row × 4 K-groups."""

    comptime V_SUB_ROWS = 8
    """V SMEM sub-block rows. Matches the BF16 reference `st_8x32_s`
    row count; FP8 V uses paired-lane `ds_read_tr8_b64` (32×32×64 path) or
    a per-lane scalar gather (16×16×128 path); both treat the V
    SMEM slab as a contiguous BN×depth block. The sub-block dims
    still drive the parent SMEM layout in `mha_prefill_v2.mojo`;
    keep `V_SUB_ROWS=8` so the cooperative DMA producer geometry
    (`SubTileLoaderLDS_st_8x32`) is dtype/shape-agnostic."""

    comptime V_SUB_COLS = 64 if Self.T.is_float8() else 32
    """V SMEM sub-block cols. 32 BF16 elts = 64 B/row; 64 FP8 elts =
    64 B/row. Byte-equivalent, same SMEM geometry for both FP8 MMA
    shapes."""

    # `ROWL_HALF_LANES` and `ROWL_STRIDE` describe the per-lane
    # decomposition of the K MFMA A-side fragment.
    # For 32×32×{16,64} (MMA_M=32): lanes split into a 32-lanes-per-row
    # × 2-cols partition; each base tile = 32 rows × MMA_K cols, and
    # each half-warp covers half the K direction.
    # For 16×16×128 (MMA_M=16): lanes split into a 16-lanes-per-row ×
    # 4-K-groups partition; each base tile = 16 rows × 128 K, and 4
    # lane-groups cover the K direction in 32-element strips.
    comptime ROWL_HALF_LANES = 16 if Self.MMA_M == 16 else 32
    """Lanes per row in the K MFMA A-side lane partition. Equal to
    MMA_M for the standard partition."""

    comptime ROWL_STRIDE = Self.MMA_K // 2 if Self.MMA_M == 32 else (
        Self.MMA_K // 4
    )
    """Per-lane K-direction fragment width.

    For 32×32×{16,64}: `MMA_K // 2`. Half-warp split packs 2 K-strips
    per base tile (8 for BF16 / 32 for FP8 32×32×64).

    For 16×16×128: `MMA_K // 4 = 32`. 4 K-groups per base tile; each
    lane-group of 16 lanes owns one K-group of 32 FP8 elements."""

    comptime FRAG_ELTS = (Self.MMA_M * Self.MMA_K) // 64
    """Input elements per lane per MFMA base tile.

    8 for BF16 (32×32×16). 32 for FP8 32×32×64. 32 for FP8 16×16×128
    (the smaller M-dim is offset by the larger K-dim, total
    M×K÷64 is identical)."""

    # Dimensional constraints (enforced in `load_K` / `load_V` /
    # `mma_QK` / `mma_PV` rather than at struct scope — Mojo allows
    # `comptime assert` only inside function bodies). The PV MFMA
    # needs `att.K = MMA_K`, so `KV_BLOCK >= MMA_K`. FP8 16×16×128
    # therefore requires `KV_BLOCK >= 128` — the layout shape
    # `KV_BLOCK / MMA_K` would otherwise floor to 0.

    # Register-tile layouts. The outer two dims count base tiles; the
    # third dim is the per-lane fragment size.

    comptime Q_LAYOUT = row_major[
        Self.Q_BLOCK_SIZE // Self.MMA_M,
        Self.DEPTH // Self.MMA_K,
        (Self.MMA_M * Self.MMA_K) // 64,
    ]()
    """Q register tile."""

    comptime K_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_M,
        Self.DEPTH // Self.MMA_K,
        (Self.MMA_M * Self.MMA_K) // 64,
    ]()
    """K register tile (whole K, pre-loaded across cluster boundaries)."""

    comptime V_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_K,
        Self.DEPTH // Self.MMA_N,
        (Self.MMA_K * Self.MMA_N) // 64,
    ]()
    """V register tile."""

    comptime ATT_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_M,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Attention block (QK output, col_l rt_32x32 FP32)."""

    comptime PV_A_FRAG_ELTS = (Self.MMA_K * Self.MMA_N) // 64
    """Per-lane PV-A fragment width = MMA_K * MMA_N / 64.

    8 for BF16 (16*32/64). 32 for FP8 32×32×64 (64*32/64).
    32 for FP8 16×16×128 (128*16/64), same as 32×32×64.

    Folds to a literal Int at type-check time because MMA_K and
    MMA_N are both comptime constants."""

    comptime ATT_BF16_SUB_LAYOUT = row_major[
        1,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        Self.PV_A_FRAG_ELTS,
    ]()
    """One PV-A subtile (MMA_K-row strip of att, input-dtype).

    BF16: 16-row strip, per-lane 8 BF16. FP8: 64-row strip, per-lane
    32 FP8."""

    comptime ATT_BF16_FULL_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_K,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        Self.PV_A_FRAG_ELTS,
    ]()
    """Full att input-dtype tile pre-cast from FP32 (indexed by
    subtile_idx to feed `mma_PV` strip-by-strip).

    BF16: 4 subtiles (KV_BLOCK=64 / MMA_K=16). FP8: 1 subtile
    (KV_BLOCK=64 / MMA_K=64)."""

    comptime O_LAYOUT = row_major[
        Self.DEPTH // Self.MMA_M,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Output accumulator (col_l rt_32x32 FP32)."""

    comptime O_T_LAYOUT = row_major[
        Self.Q_BLOCK_SIZE // Self.MMA_M,
        Self.DEPTH // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Output transpose (row_l view of the same storage as `O_LAYOUT`)."""

    # K SMEM uses a byte-level two-XOR swizzle: the pair
    # `Swizzle(1, 1, 4)` + `Swizzle(1, 0, 6)` realizes
    # `bit5 ^= bit9; bit4 ^= bit10` on the per-lane 16-byte vec index.
    # The swizzle is byte-positional, so the same XOR formula applies
    # to both BF16 (size=2, K_SUB_COLS=32 elts) and FP8 (size=1,
    # K_SUB_COLS=64 elts) — both produce 32-row × 64-B sub-blocks.
    # FP32 SMEM (unused here) would take the identity branch.

    @staticmethod
    @always_inline
    def _swizzle_K_sub(r: Int, c: Int) -> Int:
        """Returns the byte offset of `(r, c)` within a K sub-block,
        with a two-XOR same-distance swizzle applied for BF16 and FP8.

        Equivalent to `Swizzle(2, 4, 4)` at byte scope: XORs bits 8..9
        of the byte offset (the upper 2 row bits at K_SUB_COLS=64 B)
        into bits 4..5, both at distance 4. The producer side composes
        the equivalent transformation from two vec-scope swizzles
        (`Swizzle(1, 0, 4)` + `Swizzle(1, 1, 4)`): see `k_swizzle` /
        `k_swizzle2` in `MhaPrefillV2`.

        Derived to be bank-conflict-free on the reference
        32-lanes-per-row access pattern: in each 16-lane LDS access
        cycle, the permutation of bank-quadrants is `{0, 16, 32, 48, 4,
        20, 36, 52, 8, 24, 40, 56, 12, 28, 44, 60}`: a complete
        bijection of `{0, 4, 8, ..., 60}`, for both `col_offset=0`
        (lanes 0..31) and `col_offset=32` (lanes 32..63). The original
        two-distance `bit5^=bit9; bit4^=bit10` (distances 4 + 6) variant
        left a residual LDS bank conflict that this same-distance
        permutation removes.
        """
        var offset = size_of[Self.T]() * (r * Self.K_SUB_COLS + c)
        comptime if size_of[Self.T]() == 2 or size_of[Self.T]() == 1:
            # Swizzle(2, 4, 4): bits 8..9 XORed into bits 4..5.
            var swiz = ((offset >> 8) & 3) << 4
            return offset ^ swiz
        else:
            comptime assert (
                size_of[Self.T]() == 4
            ), "MhaMmaOp._swizzle_K_sub: BF16/FP8/F32 only"
            return offset

    @staticmethod
    @always_inline
    def load_K[
        layout_dst: TensorLayout,
        layout_src: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.T, layout_dst, MutUntrackedOrigin],
        src: SMemTile[Self.T, layout_src, MutAnyOrigin],
    ):
        """Loads the whole `(KV_BLOCK, DEPTH)` K tile from SMEM into the
        row_l register tile (32×MMA_K base tiles), unswizzling on the way.

        Caller must declare K SMEM with shape
        `row_major[KV_BLOCK * (DEPTH / K_SUB_COLS), K_SUB_COLS]` so the
        sub-block id linearizes via `.tile[K_SUB_ROWS, K_SUB_COLS](id, 0)`.

        BF16 path: each 32×32 sub-block holds two 32×16 base tiles
            (col parity 0 and 1). 8 BF16 elts/lane per base tile →
            one `ds_read_b128`.
        FP8 path: each 32×64 sub-block holds exactly one 32×64 base
            tile (the col-parity loop collapses). 32 FP8 elts/lane =
            32 B = two 16-B `_load_from_lds` reads, joined.

        Parameters:
            layout_dst: Register-tile layout of `dst`. Its third
                static-shape dim must equal `FRAG_ELTS`.
            layout_src: SMEM tile layout of `src`. Caller declares
                K SMEM as `row_major[KV_BLOCK * (DEPTH / K_SUB_COLS),
                K_SUB_COLS]`.

        Args:
            dst: Destination register tile for K, shape
                `(KV_BLOCK, DEPTH)` decomposed into `MMA_M × MMA_K`
                base tiles. Written unswizzled.
            src: Source SMEM tile holding the swizzled K block.
        """
        comptime assert (
            Self.T == .bfloat16 or Self.T.is_float8()
        ), "MhaMmaOp.load_K: BF16 or FP8 only"
        comptime assert (
            layout_dst.static_shape[2] == Self.FRAG_ELTS
        ), "MhaMmaOp.load_K: dst per-base-tile elts must be FRAG_ELTS"

        comptime height = layout_dst.static_shape[0]
        comptime width = layout_dst.static_shape[1]
        comptime width_st = (width * Self.MMA_K) // Self.K_SUB_COLS

        # row_l fragment: lane `lid` owns `(row, col)` in a 32×MMA_K base
        # tile at `row = lid % 32`, `col = ROWL_STRIDE * (lid // 32)`.
        # BF16: ROWL_STRIDE=8 (16 BF16 elts/row, 2 strips); FP8:
        # ROWL_STRIDE=32 (64 FP8 elts/row, 2 strips of 32 elts each).
        var lid = Int(lane_id())
        var row_offset = lid % Self.ROWL_HALF_LANES
        var col_offset = Self.ROWL_STRIDE * (lid // Self.ROWL_HALF_LANES)

        var dst_v = dst.vectorize[1, 1, Self.FRAG_ELTS]()
        comptime assert dst_v.flat_rank == 3

        comptime if Self.T == .bfloat16:
            # Per-cell body: each `(register_row, register_col)` reads
            # `subtile.ptr + swizzled_elts` directly. Kept distinct from
            # the FP8 hoist-once + `typed_imm_offset` form below, which
            # perturbs the BF16 schedule into a K/V DMA-vs-`ds_read`
            # `vmcnt` race in `MhaPrefillV2`. The FP8 paths cannot reach
            # this branch, so the BF16 schedule stays isolated from FP8
            # edits.
            var swizzled_elts_even = (
                Self._swizzle_K_sub(row_offset, col_offset) // size_of[Self.T]()
            )
            var swizzled_elts_odd = (
                Self._swizzle_K_sub(row_offset, Self.MMA_K + col_offset)
                // size_of[Self.T]()
            )

            comptime for register_row in range(height):
                comptime for register_col in range(width):
                    comptime sub_id = (
                        register_row * width_st + register_col // 2
                    )
                    comptime is_odd_col = (register_col % 2) == 1

                    var subtile = src.tile[Self.K_SUB_ROWS, Self.K_SUB_COLS](
                        sub_id, 0
                    )
                    var swizzled_elts = (
                        swizzled_elts_odd if is_odd_col else swizzled_elts_even
                    )
                    var smem_at_lane = (subtile.ptr + swizzled_elts).bitcast[
                        BFloat16
                    ]()
                    var frag = _load_from_lds[width=Self.FRAG_ELTS](
                        smem_at_lane
                    )
                    dst_v[register_row, register_col, 0] = rebind[
                        dst_v.ElementType
                    ](frag)
        elif not Self.FP8_MMA_K_128:
            # FP8 32×32×64 path: 1 base tile per sub-block, 2 × 16-B
            # reads per lane per base tile (`col_offset` and
            # `col_offset + 16`), joined into a SIMD[FP8, 32] fragment.
            #
            # Hoist the per-lane bases ONCE (lo / hi for the two 16-B
            # halves). Per-cell `sub_id` shifts become comptime
            # immediates folded into ds_read's offset:imm by the typed
            # `_load_from_lds[typed_imm_offset_bytes=...]` path.
            # Codegen-equivalent to the per-cell `subtile.ptr +
            # swizzled_elts_*` form (the AMDGPU backend recovers the same
            # `ds_read offset:imm`, K bases stay at 2 / lo+hi), but makes
            # the hoist explicit at the source level.
            comptime _HALF = Self.FRAG_ELTS // 2  # 16 FP8 = 16 B
            comptime _K_SUB_STRIDE_BYTES = (
                Self.K_SUB_ROWS * Self.K_SUB_COLS * size_of[Self.T]()
            )
            var swizzled_elts_lo = (
                Self._swizzle_K_sub(row_offset, col_offset) // size_of[Self.T]()
            )
            var swizzled_elts_hi = (
                Self._swizzle_K_sub(row_offset, col_offset + _HALF)
                // size_of[Self.T]()
            )
            var smem_base_lo = src.ptr + swizzled_elts_lo
            var smem_base_hi = src.ptr + swizzled_elts_hi

            comptime for register_row in range(height):
                comptime for register_col in range(width):
                    comptime sub_id = register_row * width_st + register_col
                    comptime sub_offset_bytes = sub_id * _K_SUB_STRIDE_BYTES

                    var lo = _load_from_lds[
                        width=_HALF, typed_imm_offset_bytes=sub_offset_bytes
                    ](smem_base_lo)
                    var hi = _load_from_lds[
                        width=_HALF, typed_imm_offset_bytes=sub_offset_bytes
                    ](smem_base_hi)
                    var frag = lo.join(hi)
                    dst_v[register_row, register_col, 0] = rebind[
                        dst_v.ElementType
                    ](frag)
        else:
            # FP8 16×16×128 path: each base tile is 16 rows × 128 K cols.
            #   register_row m_id ∈ [0, KV_BLOCK/16) — selects 16-row
            #     strip within a 32-row sub-block. m_id // 2 picks the
            #     row-direction sub-block (`sub_row`); m_id % 2 picks
            #     top-half (rows 0..15) or bottom-half (rows 16..31).
            #   register_col ∈ [0, DEPTH/128) — selects K-direction
            #     base tile.
            #
            # Lane partition (16-lanes-per-row × 4 K-groups):
            #   row_in_base = lid % 16            (0..15 row within base tile)
            #   k_group     = (lid // 16) % 4     (0..3 K-group)
            # Each K-group covers 32 FP8 elements; 4 groups span MMA_K=128.
            # 32 FP8 = 32 B → 2× ds_read_b128 (16 B each), joined.
            #
            # K-direction split across 2 SMEM sub-blocks (K_SUB_COLS=64):
            #   k_group 0,1 read from `sub_id_lo` (col 0..63 of the
            #     K-direction sub-block 0); k_group 2,3 read from
            #     `sub_id_hi` (sub-block 1).
            #   col_in_sub  = (k_group % 2) * 32   (0 or 32)
            #
            # `k_group` is a per-lane runtime value; we compute both
            # candidate sub-tile pointers and select per-lane. The
            # alternative (per-register-col runtime branch) requires
            # `if k_group < 2 else` in the inner loop — comparable code,
            # but harder for LLVM to fold. Per-lane selection by `lid`
            # bits stays clear of branches.
            var lid = Int(lane_id())
            var row_in_base = lid % 16
            var k_group = (lid // 16) % 4
            var col_in_sub = (k_group % 2) * 32  # 0 or 32

            # Per-lane runtime selector for k_sub (0 or 1): k_group/2.
            var k_sub = k_group // 2

            comptime _HALF = Self.FRAG_ELTS // 2  # 16 FP8 = 16 B

            comptime for register_row in range(height):
                comptime sub_row = register_row // 2
                comptime half_in_sub = register_row % 2
                var row_in_sub = half_in_sub * 16 + row_in_base
                var swizzled_elts_lo = (
                    Self._swizzle_K_sub(row_in_sub, col_in_sub)
                    // size_of[Self.T]()
                )
                var swizzled_elts_hi = (
                    Self._swizzle_K_sub(row_in_sub, col_in_sub + _HALF)
                    // size_of[Self.T]()
                )
                comptime for register_col in range(width):
                    # register_col indexes the K-direction MFMA base tile;
                    # at depth=128/MMA_K=128 there is exactly one. Each
                    # base tile pulls from 2 K-direction sub-blocks (lo,
                    # hi); the per-lane k_sub selects which.
                    comptime sub_id_lo = (
                        sub_row * width_st + register_col * 2 + 0
                    )
                    comptime sub_id_hi = (
                        sub_row * width_st + register_col * 2 + 1
                    )
                    var subtile_lo = src.tile[Self.K_SUB_ROWS, Self.K_SUB_COLS](
                        sub_id_lo, 0
                    )
                    var subtile_hi = src.tile[Self.K_SUB_ROWS, Self.K_SUB_COLS](
                        sub_id_hi, 0
                    )
                    var subtile_ptr = (
                        subtile_hi.ptr if k_sub == 1 else subtile_lo.ptr
                    )
                    var smem_lo = subtile_ptr + swizzled_elts_lo
                    var smem_hi = subtile_ptr + swizzled_elts_hi
                    var lo = _load_from_lds[width=_HALF](smem_lo)
                    var hi = _load_from_lds[width=_HALF](smem_hi)
                    var frag = lo.join(hi)
                    dst_v[register_row, register_col, 0] = rebind[
                        dst_v.ElementType
                    ](frag)

    @staticmethod
    @always_inline
    def load_V[
        layout_dst: TensorLayout,
        layout_src: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.T, layout_dst, MutUntrackedOrigin],
        src: SMemTile[Self.T, layout_src, MutAnyOrigin],
    ):
        """Loads the whole V tile from SMEM into the col_l register tile.

        BF16 path: `ds_read_tr16_b64_warp` over two 8×32 sub-blocks
            (top + bot) joined into an 8-elt fragment. V_LAYOUT =
            `row_major[KV_BLOCK/16, DEPTH/32, 8]`.

        FP8 path: paired-lane `ds_read_tr8_b64`: 4 reads per (i, j)
            joined into a 32-elt fragment. V_LAYOUT =
            `row_major[KV_BLOCK/64, DEPTH/32, 32]`. Per-lane
            addressing mirrors `TiledMmaLoader.load_v_fp8_strip`, but
            the SMEM linearization uses the sub-tile-major layout
            the `st_8x32` DMA writes, not a contiguous BN×depth
            slab. See the sub-tile linearization comment below.

        Caller must declare V SMEM with shape
        `row_major[KV_BLOCK * (DEPTH / V_SUB_COLS), V_SUB_COLS]`.
        The DMA producer `SubTileLoaderLDS_st_8x32` writes
        sub-tiles (V_SUB_ROWS × V_SUB_COLS) in row-major-by-sub-tile
        order: `sub_id = sub_row * (DEPTH / V_SUB_COLS) + sub_col`,
        where `sub_row = key // V_SUB_ROWS` and `sub_col =
        depth // V_SUB_COLS`. The FP8 base tile (MMA_K=64 keys,
        MMA_N=32 depth cols) spans 8 sub-tile rows (V_SUB_ROWS=8)
        in K and half a sub-tile col in N (V_SUB_COLS=64 vs
        MMA_N=32). Loader must compute the correct sub-tile id
        from the per-lane `key`; assuming contiguous depth-blocks
        catastrophically corrupts the second (and later) depth
        block at d >= V_SUB_COLS.

        Parameters:
            layout_dst: Register-tile layout of `dst`. Its third
                static-shape dim must equal `FRAG_ELTS`.
            layout_src: SMEM tile layout of `src`. Caller declares V
                SMEM as `row_major[KV_BLOCK * (DEPTH / V_SUB_COLS),
                V_SUB_COLS]`.

        Args:
            dst: Destination register tile for V, col_l layout.
                Written unswizzled.
            src: Source SMEM tile holding the V block written by
                `SubTileLoaderLDS_st_8x32` in sub-tile-major order.
        """
        comptime assert (
            Self.T == .bfloat16 or Self.T.is_float8()
        ), "MhaMmaOp.load_V: BF16 or FP8 only"
        comptime assert (
            layout_dst.static_shape[2] == Self.FRAG_ELTS
        ), "MhaMmaOp.load_V: dst per-base-tile elts must be FRAG_ELTS"

        comptime height = layout_dst.static_shape[0]
        comptime width = layout_dst.static_shape[1]

        var dst_v = dst.vectorize[1, 1, Self.FRAG_ELTS]()
        comptime assert dst_v.flat_rank == 3

        comptime if Self.T == .bfloat16:
            comptime subtiles_per_row = width
            comptime mma_shape = IndexList[3](
                Self.MMA_M, Self.MMA_N, Self.MMA_K
            )

            comptime for i in range(height):
                comptime for j in range(width):
                    comptime top_sid = (2 * i) * subtiles_per_row + j
                    comptime bot_sid = (2 * i + 1) * subtiles_per_row + j

                    var top_tile = src.tile[Self.V_SUB_ROWS, Self.V_SUB_COLS](
                        top_sid, 0
                    )
                    var bot_tile = src.tile[Self.V_SUB_ROWS, Self.V_SUB_COLS](
                        bot_sid, 0
                    )

                    var part_top = ds_read_tr16_b64_warp[mma_shape](top_tile)
                    var part_bot = ds_read_tr16_b64_warp[mma_shape](bot_tile)
                    var frag = part_top.join(part_bot)

                    dst_v[i, j, 0] = rebind[dst_v.ElementType](frag)
        elif not Self.FP8_MMA_K_128:
            # FP8 32×32×64 path: 4 paired-lane `ds_read_tr8_b64` per
            # (i, j) at key_base ∈ {0, 16, 32, 48}, joined into one
            # SIMD[FP8, 32].
            #
            # Per-lane geometry (4 hardware rows × 16 lanes/row, hw0/hw1
            # halves) matches the precompute in
            # `TiledMmaLoader.load_v_fp8_strip`.
            #
            # SMEM linearization: V SMEM is `row_major[V_SLOT_ROWS,
            # V_SUB_COLS]` with `V_SLOT_ROWS = KV_BLOCK * (DEPTH /
            # V_SUB_COLS)`. The DMA writes sub-tiles
            # (V_SUB_ROWS=8 × V_SUB_COLS=64) in row-major-by-sub-tile
            # order — for FP8 at d=128 the sequence on SMEM is:
            #   sub-tile 0: keys [0,8)   × depth [0,64)
            #   sub-tile 1: keys [0,8)   × depth [64,128)
            #   sub-tile 2: keys [8,16)  × depth [0,64)
            #   sub-tile 3: keys [8,16)  × depth [64,128)
            #   ...
            # For a per-lane (key, depth) read:
            #   sub_row = key // V_SUB_ROWS    (8 keys per sub-tile row)
            #   sub_col = depth // V_SUB_COLS  (comptime per `j`)
            #   sub_id  = sub_row * subtiles_per_row + sub_col
            #   row_in_sub = key % V_SUB_ROWS
            #   col_in_sub = depth % V_SUB_COLS
            #   slot_row   = sub_id * V_SUB_ROWS + row_in_sub
            #   byte_off   = slot_row * V_SUB_COLS + col_in_sub
            # FP8 is 1 B/elt, so element offset = byte offset.
            #
            # Hoist pattern: the per-cell `byte_offset` decomposes
            # into:
            #   - runtime per-lane part: a function of
            #     `rel_key + hw_key_shift` and `depth_base` only
            #   - comptime per-cell part: a function of `_row_offset`,
            #     `key_base`, `_sub_col`, `_d_in_sub` only
            # Hoist the runtime part as ONE base pointer per call;
            # per-cell `key_base` × `j` × comptime shifts ride in
            # `ds_read_tr8_b64`'s offset:imm slot via the typed
            # GEP-fold pattern. Codegen-equivalent to the per-cell
            # `byte_offset` computation (the AMDGPU backend recovers the
            # same `ds_read offset:imm`), but makes the
            # "ONE-base + comptime-per-cell-offset" shape explicit at
            # the source level. This hoist closes only the per-cell axis;
            # the per-call V-base count is addressed separately by
            # `precompute_v_lane_base` / `load_V_from_lane_base`.
            #
            # Algebra: with comptime baseline `B = _row_offset + key_base`
            # (multiple of 16) and runtime `R = rel_key + hw_key_shift`
            # (range 0..15):
            #   key      = B + R
            #   sub_row  = (B + R) // 8 = B//8 + R//8
            #              (since B is mult of 16, B//8 = 2 * B//16)
            #   row_in_sub = (B + R) % 8 = R % 8
            #              (since B is mult of 8)
            # So byte_offset =
            #     (B//8 + R//8) * subtiles_per_row * V_SUB_ROWS * V_SUB_COLS
            #     + sub_col * V_SUB_ROWS * V_SUB_COLS
            #     + (R%8) * V_SUB_COLS
            #     + _d_in_sub
            #     + depth_base
            # Runtime: R//8, R%8, depth_base. Comptime: B, sub_col,
            # _d_in_sub.
            comptime _MMA_M_ = Self.MMA_M  # 32
            comptime _MMA_K_ = Self.MMA_K  # 64
            comptime _V_SUB_ROWS_ = Self.V_SUB_ROWS  # 8
            comptime _V_SUB_COLS_ = Self.V_SUB_COLS  # 64
            comptime _subtiles_per_row = Self.DEPTH // _V_SUB_COLS_
            comptime assert (
                _MMA_K_ % _V_SUB_ROWS_ == 0
            ), "FP8 32x32x64 V load: MMA_K must be a multiple of V_SUB_ROWS"
            comptime assert (
                _V_SUB_COLS_ % _MMA_M_ == 0
            ), "FP8 32x32x64 V load: V_SUB_COLS must be a multiple of MMA_M"

            var lid = Int(lane_id())
            var lane_in_row = lid % 16
            var pair_idx = lane_in_row // 2
            var is_odd = lane_in_row % 2
            var row_in_warp = lid // 16
            var is_hw1 = lid // 32
            var rel_key = (pair_idx % 4) + (pair_idx // 4) * 8
            var depth_base = (row_in_warp % 2) * 16 + is_odd * 8
            var hw_key_shift = is_hw1 * 4

            # Per-lane runtime base — computed ONCE for the whole load_V
            # call. All per-cell offsets are pure comptime additions.
            var R = rel_key + hw_key_shift  # 0..15
            var R_sub_row = R // _V_SUB_ROWS_  # 0 or 1
            var R_row_in_sub = R % _V_SUB_ROWS_  # 0..7
            var v_lane_base_offset = (
                R_sub_row * _subtiles_per_row * _V_SUB_ROWS_ * _V_SUB_COLS_
                + R_row_in_sub * _V_SUB_COLS_
                + depth_base
            )
            var v_lane_base = src.ptr + v_lane_base_offset

            comptime for i in range(height):
                comptime for j in range(width):
                    # `i` indexes the K-direction base tile (`i*MMA_K`
                    # keys per strip); `j` indexes the depth tile
                    # (`j*MMA_N` cols per tile).
                    comptime _row_offset = i * _MMA_K_
                    comptime _depth_offset = j * _MMA_M_
                    comptime _sub_col = _depth_offset // _V_SUB_COLS_
                    comptime _d_in_sub = _depth_offset % _V_SUB_COLS_

                    @always_inline
                    @__parameter
                    def _load_keys[key_base: Int]() -> SIMD[Self.T, 8]:
                        # Comptime per-cell offset, derived from
                        # B = _row_offset + key_base (multiple of 16):
                        #   ((B // V_SUB_ROWS) * subtiles_per_row +
                        #    _sub_col) * V_SUB_ROWS * V_SUB_COLS
                        #   + _d_in_sub
                        comptime _B = _row_offset + key_base
                        comptime _comptime_cell_offset = (
                            (_B // _V_SUB_ROWS_)
                            * _subtiles_per_row
                            * _V_SUB_ROWS_
                            * _V_SUB_COLS_
                            + _sub_col * _V_SUB_ROWS_ * _V_SUB_COLS_
                            + _d_in_sub
                        )
                        return ds_read_tr8_b64(
                            v_lane_base + _comptime_cell_offset
                        )

                    var r0 = _load_keys[0]()
                    var r1 = _load_keys[16]()
                    var r2 = _load_keys[32]()
                    var r3 = _load_keys[48]()
                    var frag = r0.join(r1).join(r2.join(r3))
                    dst_v[i, j, 0] = rebind[dst_v.ElementType](frag)
        else:
            # FP8 16×16×128 path: per-lane scalar gather. The MFMA
            # A-operand layout for `v_mfma_scale_f32_16x16x128_f8f6f4`
            # is a permuted gather:
            #   lane_row     = lid % 16    (row in V^T base tile = depth)
            #   lane_k_group = lid // 16   (K-group 0..3, 32 keys each)
            # Each lane reads 32 contiguous FP8 keys at one depth.
            #
            # SMEM is the parent layout `row_major[V_SLOT_ROWS,
            # V_SUB_COLS]` with `V_SLOT_ROWS = KV_BLOCK * (DEPTH /
            # V_SUB_COLS)` — sub-tiles (V_SUB_ROWS=8 × V_SUB_COLS=64)
            # written by the DMA in row-major-by-sub-tile order. The
            # base tile (i, j) spans:
            #   key_range = [i*MMA_K, (i+1)*MMA_K)
            #   depth_range = [j*MMA_M, (j+1)*MMA_M)
            # Lane `(lane_row, lane_k_group)` reads 32 scalars at:
            #   depth = j*MMA_M + lane_row
            #   keys  = i*MMA_K + lane_k_group*32 .. + 32
            # Each (key, depth) maps to the sub-tile-major SMEM byte
            # via `sub_row = key // V_SUB_ROWS`, `sub_col = depth //
            # V_SUB_COLS`, and the slot row formula in the
            # 32×32×64 branch above.
            comptime _MMA_M_ = Self.MMA_M  # 16
            comptime _MMA_K_ = Self.MMA_K  # 128
            comptime _V_SUB_ROWS_ = Self.V_SUB_ROWS  # 8
            comptime _V_SUB_COLS_ = Self.V_SUB_COLS  # 64
            comptime _subtiles_per_row = Self.DEPTH // _V_SUB_COLS_

            var lid = Int(lane_id())
            var lane_row = lid % 16
            var lane_k_group = lid // 16

            var v_base = src.ptr

            comptime for i in range(height):
                comptime for j in range(width):
                    comptime _row_offset = i * _MMA_K_
                    comptime _depth_offset = j * _MMA_M_
                    var depth_in_v = _depth_offset + lane_row
                    var sub_col = depth_in_v // _V_SUB_COLS_
                    var col_in_sub = depth_in_v % _V_SUB_COLS_
                    var key_start = _row_offset + lane_k_group * 32

                    var vals = SIMD[Self.T, Self.FRAG_ELTS]()
                    for f in range(Self.FRAG_ELTS):
                        var key_idx = key_start + f
                        var sub_row = key_idx // _V_SUB_ROWS_
                        var row_in_sub = key_idx % _V_SUB_ROWS_
                        var sub_id = sub_row * _subtiles_per_row + sub_col
                        var slot_row = sub_id * _V_SUB_ROWS_ + row_in_sub
                        var byte_offset = slot_row * _V_SUB_COLS_ + col_in_sub
                        vals[f] = (v_base + byte_offset).load[width=1]()
                    dst_v[i, j, 0] = rebind[dst_v.ElementType](vals)

    # MMA dispatch — MFMA over RegTile, named by attention semantic.

    @staticmethod
    @always_inline
    def mma_QK[
        T_att: DType,
        layout_att: TensorLayout,
        layout_k: TensorLayout,
        layout_q: TensorLayout,
        //,
    ](
        mut att: RegTile[T_att, layout_att, MutUntrackedOrigin],
        mut k: RegTile[Self.T, layout_k, MutUntrackedOrigin],
        mut q: RegTile[Self.T, layout_q, MutUntrackedOrigin],
    ):
        """QK MFMA: `att += k @ q^T`. K is A (M-outer), Q is B (N-outer).

        For each output base tile `(n, m)`:
        `att[n, m] += sum_k k[n, k] * q[m, k]`.

        Parameters:
            T_att: Element data type of the attention accumulator `att`.
                FP32 for the production path.
            layout_att: Register-tile layout of `att`. Its height and
                width index output base tiles `(n, m)`.
            layout_k: Register-tile layout of `k`. Its height (M-dim)
                must equal `layout_att`'s height; its width (K-dim) must
                equal `layout_q`'s width.
            layout_q: Register-tile layout of `q`. Its height (N-dim)
                must equal `layout_att`'s width.

        Args:
            att: Output attention accumulator register tile, updated as
                `att += k @ q^T`.
            k: K register tile, the A-operand of the MFMA (M-outer).
            q: Q register tile, the B-operand of the MFMA (N-outer).
        """
        comptime assert (
            Self.T == .bfloat16 or Self.T == .float8_e4m3fn
        ), "MhaMmaOp.mma_QK: T must be bfloat16 or float8_e4m3fn"
        comptime ATT_h = layout_att.static_shape[0]
        comptime ATT_w = layout_att.static_shape[1]
        comptime K_count = layout_k.static_shape[1]

        comptime assert (
            layout_k.static_shape[1] == layout_q.static_shape[1]
        ), "mma_QK: K.K (width) != Q.K (width)"
        comptime assert (
            layout_k.static_shape[0] == ATT_h
        ), "mma_QK: K.M (height) != att.M (height)"
        comptime assert (
            layout_q.static_shape[0] == ATT_w
        ), "mma_QK: Q.N (height) != att.N (width)"

        var att_v = att.vectorize[1, 1, layout_att.static_shape[2]]()
        var k_v = k.vectorize[1, 1, layout_k.static_shape[2]]()
        var q_v = q.vectorize[1, 1, layout_q.static_shape[2]]()
        comptime assert att_v.flat_rank == 3
        comptime assert k_v.flat_rank == 3
        comptime assert q_v.flat_rank == 3

        comptime for n in range(ATT_h):
            comptime for m in range(ATT_w):
                var d_simd = att_v[n, m, 0]
                comptime for kk in range(K_count):
                    gpu_mma(d_simd, k_v[n, kk, 0], q_v[m, kk, 0], d_simd)
                att_v[n, m, 0] = d_simd

    @staticmethod
    @always_inline
    def mma_PV[
        T_o: DType,
        layout_o: TensorLayout,
        layout_v: TensorLayout,
        layout_p: TensorLayout,
        //,
    ](
        mut o: RegTile[T_o, layout_o, MutUntrackedOrigin],
        mut v: RegTile[Self.T, layout_v, MutUntrackedOrigin],
        mut p: RegTile[Self.T, layout_p, MutUntrackedOrigin],
    ):
        """PV MFMA: `o += v^T @ p`. V is A (K-outer), P is B (K-outer,
        JIT-cast to `Self.T` from `att_block`).

        For each output base tile `(n, m)`:
        `o[n, m] += sum_k v[k, n] * p[k, m]`.

        Parameters:
            T_o: Element data type of the output accumulator `o`.
                FP32 for the production path.
            layout_o: Register-tile layout of `o`. Its height and
                width index output base tiles `(n, m)`.
            layout_v: Register-tile layout of `v`. Its height (K-dim)
                must equal `layout_p`'s height; its width (M-dim) must
                equal `layout_o`'s height.
            layout_p: Register-tile layout of `p`. Its width (N-dim)
                must equal `layout_o`'s width.

        Args:
            o: Output accumulator register tile, updated as
                `o += v^T @ p`.
            v: V register tile, the A-operand of the MFMA (K-outer).
            p: P register tile, the B-operand of the MFMA (K-outer,
                JIT-cast to `Self.T` from `att_block`).
        """
        comptime assert (
            Self.T == .bfloat16 or Self.T == .float8_e4m3fn
        ), "MhaMmaOp.mma_PV: T must be bfloat16 or float8_e4m3fn"
        comptime O_h = layout_o.static_shape[0]
        comptime O_w = layout_o.static_shape[1]
        comptime K_count = layout_v.static_shape[0]

        comptime assert (
            layout_v.static_shape[0] == layout_p.static_shape[0]
        ), "mma_PV: V.K (height) != P.K (height)"
        comptime assert (
            layout_v.static_shape[1] == O_h
        ), "mma_PV: V.M (width) != o.M (height)"
        comptime assert (
            layout_p.static_shape[1] == O_w
        ), "mma_PV: P.N (width) != o.N (width)"

        var o_v = o.vectorize[1, 1, layout_o.static_shape[2]]()
        var v_v = v.vectorize[1, 1, layout_v.static_shape[2]]()
        var p_v = p.vectorize[1, 1, layout_p.static_shape[2]]()
        comptime assert o_v.flat_rank == 3
        comptime assert v_v.flat_rank == 3
        comptime assert p_v.flat_rank == 3

        comptime for n in range(O_h):
            comptime for m in range(O_w):
                var d_simd = o_v[n, m, 0]
                comptime for kk in range(K_count):
                    gpu_mma(d_simd, v_v[kk, n, 0], p_v[kk, m, 0], d_simd)
                o_v[n, m, 0] = d_simd

    @staticmethod
    @always_inline
    def exp2_inplace_range[
        T_att: DType,
        layout: TensorLayout,
        //,
        start: Int,
        end: Int,
    ](mut tile: RegTile[T_att, layout, MutUntrackedOrigin],):
        """In-place `exp2` over a base-tile-aligned per-lane slice
        `tile[start:end]`. `start` and `end` must be multiples of the
        fragment width so the slice maps to whole base tiles. Used to
        split first / second half of the softmax exp2 across PV MFMAs.

        `T_att` is FP32 in the BF16 attention path and BF16 in the FP8
        attention path's BF16 softmax. `math_exp2` works with either;
        for BF16 it lowers to `v_cvt_f32_bf16` +
        `v_exp_f32` + `v_cvt_pkrtz_bf16_f32` (no packed transcendental
        on gfx950, `v_exp_f32` is scalar regardless of input dtype).

        Parameters:
            T_att: Element data type of the attention tile. FP32 in the
                BF16 attention path; BF16 in the FP8 attention path's BF16
                softmax.
            layout: Register-tile layout of `tile`. Its third static-shape
                dim is the per-lane fragment width that `start` and `end`
                must align to.
            start: Starting per-lane element index of the slice,
                inclusive. Must be a multiple of the fragment width and lie
                in `[0, total)`.
            end: Ending per-lane element index of the slice, exclusive.
                Must be a multiple of the fragment width and lie in
                `[start, total)`.

        Args:
            tile: Attention register tile exponentiated in place over the
                `[start, end)` per-lane slice.
        """
        comptime assert (
            T_att.is_floating_point()
        ), "exp2_inplace_range: T_att must be floating-point"
        comptime _W = layout.static_shape[1]
        comptime _F = layout.static_shape[2]
        comptime _N = layout.static_shape[0] * _W * _F
        comptime assert (
            0 <= start <= end <= _N
        ), "exp2_inplace_range: [start, end) must lie in [0, total)"
        comptime assert (
            start % _F == 0 and end % _F == 0
        ), "exp2_inplace_range: start/end must be multiples of fragment width"
        var v = tile.vectorize[1, 1, _F]()
        comptime assert v.flat_rank == 3
        comptime for b in range(start // _F, end // _F):
            comptime _h, _w = divmod(b, _W)
            v[_h, _w, 0] = math_exp2(v[_h, _w, 0])


# ===----------------------------------------------------------------------=== #
# MLA MMA op — `MhaMmaOp` extended with the MLA FP8 fragment loaders.
# ===----------------------------------------------------------------------=== #


struct MlaMmaOp[T: DType, config: MhaConfigV2]:
    """MLA MMA op for `MlaPrefillV2` / `MlaPrefillV2Core`.

    Extends `MhaMmaOp` (re-exporting its shape constants / register-tile
    layouts and delegating `_swizzle_K_sub` / `mma_QK` / `mma_PV` /
    `exp2_inplace_range`) with the MLA FP8 32x32x64 fragment loaders
    (`load_K_frag`, `precompute_v_lane_base`, `load_V_from_lane_base`,
    `load_V_frag`) that stream K/V through a rotating in-register band
    (the reference never materializes the whole V tile).

    The split isolates these FP8 extensions from MHA: the
    `MhaPrefillV2` BF16 prefill path uses `MhaMmaOp` directly, so an edit
    to MLA's FP8 fragment loaders can no longer perturb the MHA kernel's
    BF16 `load_K` / `load_V` schedule (interleaving FP8 branches into the
    shared whole-tile loaders opens a K/V DMA-vs-`ds_read` `vmcnt` race
    in `MhaPrefillV2`).

    MLA never calls the whole-tile `load_K` / `load_V`; it consumes
    K/V fragment-at-a-time via the `*_frag` loaders. The whole-tile
    loaders therefore stay on `MhaMmaOp` only.

    Parameters:
        T: Element data type (FP8 e4m3 for the MLA prefill path).
        config: Shape configuration (`MhaConfigV2`, via `MlaConfigV2.mha()`).
    """

    comptime _Shared = MhaMmaOp[Self.T, Self.config]

    # --- Shape constants / register-tile layouts: re-exported from the
    #     shared MHA op so the producer/consumer geometry is identical
    #     and `type_of(_MmaOp.X)` resolves to the same layout types.
    comptime Q_BLOCK_SIZE = Self._Shared.Q_BLOCK_SIZE
    comptime KV_BLOCK = Self._Shared.KV_BLOCK
    comptime DEPTH = Self._Shared.DEPTH
    comptime FP8_MMA_K_128 = Self._Shared.FP8_MMA_K_128
    comptime MMA_M = Self._Shared.MMA_M
    comptime MMA_N = Self._Shared.MMA_N
    comptime MMA_K = Self._Shared.MMA_K
    comptime K_SUB_ROWS = Self._Shared.K_SUB_ROWS
    comptime K_SUB_COLS = Self._Shared.K_SUB_COLS
    comptime V_SUB_ROWS = Self._Shared.V_SUB_ROWS
    comptime V_SUB_COLS = Self._Shared.V_SUB_COLS
    comptime ROWL_HALF_LANES = Self._Shared.ROWL_HALF_LANES
    comptime ROWL_STRIDE = Self._Shared.ROWL_STRIDE
    comptime FRAG_ELTS = Self._Shared.FRAG_ELTS
    comptime Q_LAYOUT = Self._Shared.Q_LAYOUT
    comptime K_LAYOUT = Self._Shared.K_LAYOUT
    comptime V_LAYOUT = Self._Shared.V_LAYOUT
    comptime ATT_LAYOUT = Self._Shared.ATT_LAYOUT
    comptime PV_A_FRAG_ELTS = Self._Shared.PV_A_FRAG_ELTS
    comptime ATT_BF16_SUB_LAYOUT = Self._Shared.ATT_BF16_SUB_LAYOUT
    comptime ATT_BF16_FULL_LAYOUT = Self._Shared.ATT_BF16_FULL_LAYOUT
    comptime O_LAYOUT = Self._Shared.O_LAYOUT
    comptime O_T_LAYOUT = Self._Shared.O_T_LAYOUT

    # --- Shared swizzle / MFMA / softmax methods: delegate to `MhaMmaOp`.

    @staticmethod
    @always_inline
    def _swizzle_K_sub(r: Int, c: Int) -> Int:
        """K sub-block byte swizzle: delegates to `MhaMmaOp`."""
        return Self._Shared._swizzle_K_sub(r, c)

    @staticmethod
    @always_inline
    def mma_QK[
        T_att: DType,
        layout_att: TensorLayout,
        layout_k: TensorLayout,
        layout_q: TensorLayout,
        //,
    ](
        mut att: RegTile[T_att, layout_att, MutUntrackedOrigin],
        mut k: RegTile[Self.T, layout_k, MutUntrackedOrigin],
        mut q: RegTile[Self.T, layout_q, MutUntrackedOrigin],
    ):
        """QK MFMA: delegates to `MhaMmaOp.mma_QK` (body identical).

        Parameters:
            T_att: Element data type of the attention accumulator `att`.
                FP32 for the production path.
            layout_att: Register-tile layout of `att`. Its height and
                width index output base tiles `(n, m)`.
            layout_k: Register-tile layout of `k`. Its height (M-dim)
                must equal `layout_att`'s height; its width (K-dim) must
                equal `layout_q`'s width.
            layout_q: Register-tile layout of `q`. Its height (N-dim)
                must equal `layout_att`'s width.

        Args:
            att: Output attention accumulator register tile, updated as
                `att += k @ q^T`.
            k: K register tile, the A-operand of the MFMA (M-outer).
            q: Q register tile, the B-operand of the MFMA (N-outer).
        """
        Self._Shared.mma_QK(att, k, q)

    @staticmethod
    @always_inline
    def mma_PV[
        T_o: DType,
        layout_o: TensorLayout,
        layout_v: TensorLayout,
        layout_p: TensorLayout,
        //,
    ](
        mut o: RegTile[T_o, layout_o, MutUntrackedOrigin],
        mut v: RegTile[Self.T, layout_v, MutUntrackedOrigin],
        mut p: RegTile[Self.T, layout_p, MutUntrackedOrigin],
    ):
        """PV MFMA: delegates to `MhaMmaOp.mma_PV` (body identical).

        Parameters:
            T_o: Element data type of the output accumulator `o`. FP32
                for the production path.
            layout_o: Register-tile layout of `o`. Its height and width
                index output base tiles `(n, m)`.
            layout_v: Register-tile layout of `v`. Its height (K-dim)
                must equal `layout_p`'s height; its width (M-dim) must
                equal `layout_o`'s height.
            layout_p: Register-tile layout of `p`. Its width (N-dim)
                must equal `layout_o`'s width.

        Args:
            o: Output accumulator register tile, updated as
                `o += v^T @ p`.
            v: V register tile, the A-operand of the MFMA (K-outer).
            p: P register tile, the B-operand of the MFMA (K-outer,
                JIT-cast to `Self.T` from `att_block`).
        """
        Self._Shared.mma_PV(o, v, p)

    @staticmethod
    @always_inline
    def exp2_inplace_range[
        T_att: DType,
        layout: TensorLayout,
        //,
        start: Int,
        end: Int,
    ](mut tile: RegTile[T_att, layout, MutUntrackedOrigin],):
        """`exp2` over a slice: delegates to `MhaMmaOp.exp2_inplace_range`.

        Parameters:
            T_att: Element data type of the attention tile. FP32 in the
                BF16 attention path; BF16 in the FP8 attention path's BF16
                softmax.
            layout: Register-tile layout of `tile`. Its third static-shape
                dim is the per-lane fragment width that `start` and `end`
                must align to.
            start: Starting per-lane element index of the slice,
                inclusive. Must be a multiple of the fragment width and lie
                in `[0, total)`.
            end: Ending per-lane element index of the slice, exclusive.
                Must be a multiple of the fragment width and lie in
                `[start, total)`.

        Args:
            tile: Attention register tile exponentiated in place over the
                `[start, end)` per-lane slice.
        """
        Self._Shared.exp2_inplace_range[start, end](tile)

    # --- FP8 32x32x64 fragment loaders (MLA-only). ---

    @staticmethod
    @always_inline
    def load_K_frag[
        sub_id: Int,
    ](src: SMemTile[Self.T, _, MutAnyOrigin],) -> SIMD[Self.T, Self.FRAG_ELTS]:
        """Loads ONE K MFMA fragment (`sub_id`) from the K SMEM sub-view
        `src` and returns it as a SIMD value: the single-fragment
        factoring of the FP8 32x32x64 K-load inner loop.

        Returning a SIMD value (rather than writing a register tile)
        lets the caller stream fragments through a rotating in-register
        band where each slot is a plain SSA value, sidestepping the
        strided-register-sub-view write that lands on the wrong VGPRs
        when a sub-tile of a larger `reg_alloc` is the destination at a
        non-zero offset. The fragment feeds `gpu_mma` directly (the QK
        A-operand).

        `sub_id` is the K SMEM sub-block index; it enters only as a
        comptime `ds_read offset:` immediate. The two per-lane swizzled
        bases are loop-invariant in `src.ptr`, so across a caller's
        unrolled per-fragment stream LLVM CSEs them to a single base
        register pair (the reference's `v226` single-base K pattern).

        Constraints:
            FP8 32x32x64 path only (`MMA_K == 64`).

        Parameters:
            sub_id: K SMEM sub-block index, entering only as a comptime
                `ds_read offset:` immediate.

        Args:
            src: K SMEM sub-view holding the swizzled K block, read at
                the per-lane swizzled offset for `sub_id`.
        """
        comptime assert (
            Self.T.is_float8() and not Self.FP8_MMA_K_128
        ), "MlaMmaOp.load_K_frag: FP8 32x32x64 only"
        comptime _HALF = Self.FRAG_ELTS // 2  # 16 FP8 = 16 B
        comptime _K_SUB_STRIDE_BYTES = (
            Self.K_SUB_ROWS * Self.K_SUB_COLS * size_of[Self.T]()
        )
        comptime sub_offset_bytes = sub_id * _K_SUB_STRIDE_BYTES
        var lid = Int(lane_id())
        var row_offset = lid % Self.ROWL_HALF_LANES
        var col_offset = Self.ROWL_STRIDE * (lid // Self.ROWL_HALF_LANES)
        var swizzled_elts_lo = (
            Self._swizzle_K_sub(row_offset, col_offset) // size_of[Self.T]()
        )
        var swizzled_elts_hi = (
            Self._swizzle_K_sub(row_offset, col_offset + _HALF)
            // size_of[Self.T]()
        )
        var lo = _load_from_lds[
            width=_HALF, typed_imm_offset_bytes=sub_offset_bytes
        ](src.ptr + swizzled_elts_lo)
        var hi = _load_from_lds[
            width=_HALF, typed_imm_offset_bytes=sub_offset_bytes
        ](src.ptr + swizzled_elts_hi)
        # `join` yields SIMD[T, 2*_HALF]; rebind to the FRAG_ELTS form
        # (`(MMA_M*MMA_K)//64`) the caller's tile element expects — same
        # in-memory width, syntactically distinct parameter expression.
        return rebind[SIMD[Self.T, Self.FRAG_ELTS]](lo.join(hi))

    @staticmethod
    @always_inline
    def precompute_v_lane_base[
        origin: Origin,
        //,
        v_full_v227: Bool = False,
        v227_layout: Bool = False,
    ](
        v_slot_ptr: UnsafePointer[
            Scalar[Self.T], origin, address_space=.SHARED
        ],
    ) -> UnsafePointer[Scalar[Self.T], origin, address_space=.SHARED]:
        """Computes the per-lane V LDS base pointer for the FP8 32x32x64
        path ONCE per V SMEM slot (caller passes `v_smem_<stage>.ptr`).
        Hoisting this base out of the per-fragment readout collapses the
        per-call lane offset across all invocations from the same slot
        into ONE shared materialization, mirroring the reference's
        `v227` base carried across the entire main loop. Caller threads
        the returned pointer into `load_V_frag`. Origin is propagated
        from the input slot so the returned pointer carries the slot's
        lifetime/mutability annotation.

        Parameters:
            origin: Pointer origin (lifetime / mutability provenance) of the
                V SMEM slot, propagated to the returned pointer so it carries
                the slot's annotation (inferred).
            v_full_v227: Reference `v227` V adapter per-lane READ base
                (Bool). Default False → byte-identical. When True, replaces
                the per-lane base ENTIRELY with the reference's exact
                `v227` formula (the WHOLE read map), a per-tr8-cycle
                bank-quadrant bijection that eliminates the V-transpose
                LDS bank conflict. This is HALF of the adapter `R`: the
                caller MUST also pass `v_full_v227=True` to `load_V_frag`
                (the faithful readout cell) AND run the matching producer
                (`SubTileLoaderLDS_st_8x32[v_full_v227=True]` /
                `MlaPrefillV2Core._dma_v[v_full_v227=True]`, the `W`); the three
                compose to the standard PV fragment (numerically
                equivalent). FP8 32x32x64 only. The default-on V LDS
                adapter for `MlaPrefillV2`; production MLA passes
                `v_full_v227=False`. `-D v_full_v227`.
            v227_layout: Spell the `v_full_v227` per-lane READ base via CuTe
                Layout Algebra (`crd2idx` over a per-bit `Coord`) instead of
                the hand-rolled runtime bit arithmetic (Bool, only consulted
                when `v_full_v227` is True). SAME mapping, different spelling,
                the `v227` base is bit-LINEAR over the bit-decomposed lane,
                so it is `crd2idx(lane, Coord(2,2,2,2,2,2), Coord(8, 0x80,
                0x820, 0x1040, 16, 0x410))` (the 2-bit field `((v0>>2)&3)*
                0x820` splits to bit2*0x820 + bit3*0x1040). Mirrors the WRITE
                side (`SubTileLoaderLDS_st_8x32[_v227_layout]`); this is
                the no-underscore parameter form of the WRITE side's struct
                field `_v227_layout`, both driven by `-D v227_layout`.
                Numerically equivalent. `-D v227_layout`.

        Args:
            v_slot_ptr: Base pointer of the V SMEM slot for a given stage
                (caller passes `v_smem_<stage>.ptr`). The per-lane read base
                is computed as an offset from this pointer.
        """
        comptime assert (
            Self.T.is_float8() and not Self.FP8_MMA_K_128
        ), "MlaMmaOp.precompute_v_lane_base: FP8 32x32x64 path only"
        comptime _V_SUB_ROWS_ = Self.V_SUB_ROWS  # 8
        comptime _V_SUB_COLS_ = Self.V_SUB_COLS  # 64
        comptime _subtiles_per_row = Self.DEPTH // _V_SUB_COLS_
        comptime _v_block_stride = (
            _subtiles_per_row * _V_SUB_ROWS_ * _V_SUB_COLS_
        )

        var lid = Int(lane_id())
        var lane_in_row = lid % 16
        var pair_idx = lane_in_row // 2
        var is_odd = lane_in_row % 2
        var row_in_warp = lid // 16
        var is_hw1 = lid // 32
        var rel_key = (pair_idx % 4) + (pair_idx // 4) * 8
        var depth_base = (row_in_warp % 2) * 16 + is_odd * 8
        var hw_key_shift = is_hw1 * 4

        var R = rel_key + hw_key_shift  # 0..15
        var R_sub_row = R // _V_SUB_ROWS_  # 0 or 1
        var R_row_in_sub = R % _V_SUB_ROWS_  # 0..7
        var v_lane_base_offset = (
            R_sub_row * _v_block_stride
            + R_row_in_sub * _V_SUB_COLS_
            + depth_base
        )
        # Reference `v227` V adapter per-lane READ base (the per-lane half
        # of the adapter `R`). Replaces OUR per-lane base decode entirely
        # with the reference's EXACT `v227` formula:
        #   v227 = ((v0>>1)&1)*0x80 + ((v0>>2)&3)*0x820 + (v0&1)*8
        #          + ((v0>>4)&1)*16 + ((v0>>5)&1)*0x410
        # (the leading LDS region base is already in `v_slot_ptr`). v227's
        # 16-lanes-per-tr8-cycle tile 32 distinct bank-quadrants (max reuse
        # 1) so the transpose read is conflict-free; OURS tile only 16 (max
        # reuse 2). The matching producer
        # (`SubTileLoaderLDS_st_8x32[v_full_v227=True]`) writes V in the
        # chunk-stepped layout this base reads back, and `load_V_frag
        # [v_full_v227=True]` adds the faithful per-cell offset — the three
        # compose to the standard PV fragment (numerically equivalent;
        # the write `W` is `pi o W_ours`, the LDS-byte permutation `pi:
        # ours_read_addr -> ref_read_addr` proven a bijection over the
        # slot).
        comptime if v_full_v227:
            comptime if v227_layout:
                # ----- Layout-Algebra expression of the SAME `v227` base
                # (`-D v227_layout`, the default for MlaPrefillV2) ---
                # The v227 per-lane base is BIT-LINEAR over the bit-decomposed
                # lane, so it is `crd2idx` against a per-bit `Coord` whose
                # strides carry the per-bit weights declaratively (the bit-
                # DECOMPOSITION, idx2crd over shape (2,...), stays explicit;
                # the Coord names only the stride table). Per lane bit:
                #   bit0 -> 8     bit1 -> 0x80   bit2 -> 0x820
                #   bit3 -> 0x1040  bit4 -> 16   bit5 -> 0x410
                # bits 2,3 reproduce the hand `((v0>>2)&3)*0x820` =
                # bit2*0x820 + bit3*(2*0x820=0x1040). Bit-exact over
                # lanes 0..63 against the hand formula; numerically
                # equivalent. Mirrors the WRITE side's `_v227_layout` spelling
                # (a `Scalar` leaf is the runtime-coord form; the comptime
                # `Coord(...)` shape/stride drive the free `crd2idx`).
                comptime _V227_BASE_SHAPE = Coord(2, 2, 2, 2, 2, 2)
                comptime _V227_BASE_STRIDE = Coord(
                    8, 0x80, 0x820, 0x1040, 16, 0x410
                )
                v_lane_base_offset = Int(
                    crd2idx(
                        Int32(lid),
                        _V227_BASE_SHAPE,
                        _V227_BASE_STRIDE,
                    )
                )
            else:
                # ----- DEFAULT: hand-rolled runtime bit arithmetic -----------
                var v0 = lid
                v_lane_base_offset = (
                    ((v0 >> 1) & 1) * 0x80
                    + ((v0 >> 2) & 3) * 0x820
                    + (v0 & 1) * 8
                    + ((v0 >> 4) & 1) * 16
                    + ((v0 >> 5) & 1) * 0x410
                )
        # Propagates the slot's origin via the `origin` parameter, so the
        # returned pointer carries the input slot's lifetime/mutability and
        # the caller's two per-slot bases share one concrete type.
        return v_slot_ptr + v_lane_base_offset

    @staticmethod
    @always_inline
    def load_V_from_lane_base[
        layout_dst: TensorLayout,
    ](
        mut dst: RegTile[Self.T, layout_dst, MutUntrackedOrigin],
        v_lane_base: UnsafePointer[Scalar[Self.T], _, address_space=.SHARED],
    ):
        """FP8 32x32x64 V load consuming a pre-computed per-lane base
        pointer (hoisted by the caller). Each call site adds ONLY
        comptime per-cell offsets to `v_lane_base`: no per-call
        per-lane base recomputation, no `src.ptr` indirection.

        Collapses the 27 distinct V ds_read base VGPRs at KV=128 (one
        per V-load × slot toggle) toward the reference's ~3-base pattern
        (one per SMEM slot, carried across all calls from that slot).
        Codegen-equivalent to the equivalent in-call recomputation
        on the K=128 fast path (the AMDGPU backend recovers the same
        `ds_read offset:imm`) but makes the kernel-level hoist explicit at
        the source level.

        Parameters:
            layout_dst: Register-tile layout of `dst`. Its third
                static-shape dim must equal `FRAG_ELTS`.

        Args:
            dst: Destination register tile for V, col_l layout. Written
                unswizzled.
            v_lane_base: Pre-computed per-lane V LDS base pointer (from
                `precompute_v_lane_base`), carrying the V SMEM slot's
                origin. Each call adds only comptime per-cell offsets to
                this base.
        """
        comptime assert (
            Self.T.is_float8() and not Self.FP8_MMA_K_128
        ), "MlaMmaOp.load_V_from_lane_base: FP8 32x32x64 path only"
        comptime assert layout_dst.static_shape[2] == Self.FRAG_ELTS, (
            "MlaMmaOp.load_V_from_lane_base: dst per-base-tile elts"
            " must be FRAG_ELTS"
        )

        comptime height = layout_dst.static_shape[0]
        comptime width = layout_dst.static_shape[1]

        var dst_v = dst.vectorize[1, 1, Self.FRAG_ELTS]()
        comptime assert dst_v.flat_rank == 3

        comptime _MMA_M_ = Self.MMA_M  # 32
        comptime _MMA_K_ = Self.MMA_K  # 64
        comptime _V_SUB_ROWS_ = Self.V_SUB_ROWS  # 8
        comptime _V_SUB_COLS_ = Self.V_SUB_COLS  # 64
        comptime _subtiles_per_row = Self.DEPTH // _V_SUB_COLS_
        comptime _v_block_stride = (
            _subtiles_per_row * _V_SUB_ROWS_ * _V_SUB_COLS_
        )

        comptime for i in range(height):
            comptime for j in range(width):
                comptime _row_offset = i * _MMA_K_
                comptime _depth_offset = j * _MMA_M_
                comptime _sub_col = _depth_offset // _V_SUB_COLS_
                comptime _d_in_sub = _depth_offset % _V_SUB_COLS_

                @always_inline
                @__parameter
                def _load_keys[key_base: Int]() -> SIMD[Self.T, 8]:
                    comptime _B = _row_offset + key_base
                    comptime _comptime_cell_offset = (
                        (_B // _V_SUB_ROWS_) * _v_block_stride
                        + _sub_col * _V_SUB_ROWS_ * _V_SUB_COLS_
                        + _d_in_sub
                    )
                    return ds_read_tr8_b64(v_lane_base + _comptime_cell_offset)

                var r0 = _load_keys[0]()
                var r1 = _load_keys[16]()
                var r2 = _load_keys[32]()
                var r3 = _load_keys[48]()
                var frag = r0.join(r1).join(r2.join(r3))
                dst_v[i, j, 0] = rebind[dst_v.ElementType](frag)

    @staticmethod
    @always_inline
    def load_V_frag[
        i_strip: Int,
        j_depth: Int,
        v_full_v227: Bool = False,
    ](
        v_lane_base: UnsafePointer[Scalar[Self.T], _, address_space=.SHARED],
    ) -> SIMD[Self.T, Self.FRAG_ELTS]:
        """Loads ONE V MFMA fragment `(i_strip, j_depth)` from the
        pre-computed per-lane V LDS base and returns it as a SIMD value:
        the single-fragment factoring of `load_V_from_lane_base`'s FP8
        32x32x64 inner `(i, j)` cell (the 4 paired-lane `ds_read_tr8_b64`
        joined into one `SIMD[FP8, 32]`).

        Returning a SIMD value (rather than writing a register tile) is
        the V counterpart of `load_K_frag`; it lets the caller stream V
        fragments through the SAME rotating in-register band K streamed
        through (K and V have disjoint lifetimes: K is consumed in the QK
        MFMAs before softmax, V in the PV MFMAs after, so the band's
        registers are free for V by PV time). Each slot is then a plain
        SSA value, sidestepping the strided-register-sub-view write that
        lands on the wrong VGPRs at a non-zero offset, and avoids
        materializing the whole 64-VGPR `V_LAYOUT` tile held live across
        the softmax/exp/FP8 clusters (the reference never materializes V;
        it transpose-reads V fragment-at-a-time through its reused
        `v[28:59]` band). The fragment feeds `mma_PV` / `gpu_mma`
        directly (the PV A-operand).

        `i_strip` is the K-direction base tile (`i_strip*MMA_K` keys per
        strip; `0 .. KV_BLOCK/MMA_K`); `j_depth` is the depth tile
        (`j_depth*MMA_N` cols; `0 .. DEPTH/MMA_N`). Both enter only as
        comptime `ds_read_tr8_b64 offset:` immediates, so across a
        caller's unrolled per-fragment stream LLVM CSEs `v_lane_base` to
        a single base-register set (the reference's `v227` single-base V
        pattern), the same hoist `load_V_from_lane_base` already relies
        on.

        Constraints:
            FP8 32x32x64 path only (`MMA_K == 64`).

        Parameters:
            i_strip: K-direction base tile index.
            j_depth: Depth tile index.
            v_full_v227: Reference `v227` V adapter READ cell (Bool).
                Default False → OUR `st_8x32` cell decode (byte-identical).
                When True, uses the faithful reference readout cell offset
                `i_strip*0x2080 + j_depth*0x20 + r*0x100` (r = subread 0..3)
                on top of the `v227` per-lane base
                (`precompute_v_lane_base[v_full_v227=True]`). This is the
                `R` of the adapter `W∘R` pair; the producer
                (`SubTileLoaderLDS_st_8x32[v_full_v227=True]` /
                `MlaPrefillV2Core._dma_v[v_full_v227=True]`) MUST set
                `v_full_v227=True` too, or V scrambles. The two compose to
                the standard PV fragment,                 bank-conflict-free (`v227`'s
                per-tr8-cycle bank-quadrant bijection). FP8 32x32x64 only.

        Args:
            v_lane_base: Pre-computed per-lane V LDS base pointer (from
                `precompute_v_lane_base`), carrying the V SMEM slot's
                origin. Each call adds only comptime per-cell offsets to
                this base.
        """
        comptime assert (
            Self.T.is_float8() and not Self.FP8_MMA_K_128
        ), "MlaMmaOp.load_V_frag: FP8 32x32x64 only"

        comptime _MMA_M_ = Self.MMA_M  # 32
        comptime _MMA_K_ = Self.MMA_K  # 64
        comptime _V_SUB_ROWS_ = Self.V_SUB_ROWS  # 8
        comptime _V_SUB_COLS_ = Self.V_SUB_COLS  # 64
        comptime _subtiles_per_row = Self.DEPTH // _V_SUB_COLS_
        comptime _v_block_stride = (
            _subtiles_per_row * _V_SUB_ROWS_ * _V_SUB_COLS_
        )

        comptime _row_offset = i_strip * _MMA_K_
        comptime _depth_offset = j_depth * _MMA_M_
        comptime _sub_col = _depth_offset // _V_SUB_COLS_
        comptime _d_in_sub = _depth_offset % _V_SUB_COLS_

        @always_inline
        @__parameter
        def _load_keys[key_base: Int]() -> SIMD[Self.T, 8]:
            comptime _B = _row_offset + key_base
            # Reference FULL-v227 V adapter READ cell offset (the `R` of
            # the `W∘R` pair). The per-lane base `v_lane_base` is already
            # the reference's `v227` (from
            # `precompute_v_lane_base[v_full_v227=True]`); this is the
            # faithful readout cell `i_strip*0x2080 + j_depth*0x20 +
            # r*0x100` (r = key_base/16 = subread 0..3), verified against
            # `aiter_real.s` and bit-exact composing with the producer's
            # `W` over the whole slot. Replaces OUR `st_8x32` cell decode
            # entirely. Default False → byte-identical immediates.
            comptime _comptime_cell_offset = (
                i_strip * 0x2080 + j_depth * 0x20 + (key_base // 16) * 0x100
            ) if v_full_v227 else (
                (_B // _V_SUB_ROWS_) * _v_block_stride
                + _sub_col * _V_SUB_ROWS_ * _V_SUB_COLS_
                + _d_in_sub
            )
            return ds_read_tr8_b64(v_lane_base + _comptime_cell_offset)

        var r0 = _load_keys[0]()
        var r1 = _load_keys[16]()
        var r2 = _load_keys[32]()
        var r3 = _load_keys[48]()
        # `join` yields SIMD[T, 32]; rebind to the FRAG_ELTS form the
        # caller's tile element expects — same in-memory width.
        return rebind[SIMD[Self.T, Self.FRAG_ELTS]](
            r0.join(r1).join(r2.join(r3))
        )
