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
"""MLA-prefill math components for AMD MI355X (gfx950).

`MlaPrefillV2`-owned home of the MLA-prefill numeric closure that
`mla_prefill_v2.mojo` consumes: the `MlaPrefillV2Core[config]` struct
(trimmed to the methods/constants `_attend_exact` and the
host launcher actually reference), the `_MlaKDmaPair` single-base K DMA
helper, and the module-level scheduling primitives (`_sched_barrier_zero`,
`_s_barrier_raw`, `_s_setprio`).

The MLA-prefill MATH (QK with nope d=128 + rope d=64; FlashAttention-2
online softmax via `OnlineSoftmax`; in-place FP8 P collapse; PV
accumulate; normalize + store; causal / null mask) lives here so the
inner loop can reuse the verified primitives while emitting its own
cluster cadence in `mla_prefill_v2.mojo`. The shared building blocks
(`MhaMmaOp`/`MlaConfigV2` from `mha_mma_op.mojo`, the `OnlineSoftmax`
recurrence, the `MaskApplier` mask dispatch, the `SubTileLoaderLDS_*`
loaders from `amd_tile_io.mojo`) are imported in place; this module owns
only the MLA-prefill-specific assembly.

`MlaPrefillV2Core`'s `_FP32_SOFTMAX_SCORES` gate (FP8 + KV>=128 + 32x32x64) is
the default-on path for the FP8 / KV>=128 / 32x32x64 shape, so every
reused primitive exercises the codegen `MlaPrefillV2` ships.
"""

from max.gpu import WARP_SIZE, lane_id
from std.sys.intrinsics import llvm_intrinsic

from layout import TensorLayout, TileTensor, DefaultEngine
from layout._utils import make_amd_buffer_resource
from layout.coord import Coord
from layout.swizzle import Swizzle
from layout.tile_layout import (
    ComptimeInt,
    Idx,
    Layout as TileLayout,
    col_major,
    row_major,
)
from structured_kernels.amd_tile_io import (
    RegTile,
    RegTileEpilogue,
    RegTileLoader,
    SMemTile,
    SubTileLoaderLDS,
    SubTileLoaderLDS_st_8x32,
    reg_alloc,
)

from nn.attention.mha_operand import MHAOperand

from .buffers import _cast_f32_to_fp8_raw
from .mha_softmax import OnlineSoftmax
from .mha_mma_op import MlaConfigV2, MlaMmaOp
from .iglp import sched_barrier_exp_pairs


# ==--------------------------------------------------------------------==
# Module-level scheduling helpers. Mirror the ones in `mha_prefill_v2.mojo`
# verbatim; keep them in lockstep here so the MHA prefill path stays
# byte-stable.
# ==--------------------------------------------------------------------==


@always_inline
def _s_setprio[priority: Int16]():
    """Sets MFMA wave instruction priority (0 = normal, 1 = high)."""
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)


@always_inline
def _sched_barrier_zero():
    """`sched_barrier(0)`: hard reordering barrier."""
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@always_inline
def _s_barrier_raw():
    """Bare `s_barrier`, no implicit waitcnts so DMAs can cross."""
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


struct MlaPrefillV2Core[config: MlaConfigV2]:
    """8-warp MLA forward kernel parameterized by `MlaConfigV2`.

    Sibling of `MhaPrefillV2`. Each block runs `config.num_warps`
    wave64 warps that share K_nope / K_rope / V SMEM via cooperative
    DMA. Warp `w` owns Q rows `[w * q_block_size, (w + 1) *
    q_block_size)` of the block's stripe and carries its own
    register-resident attention state.

    Parameters:
        config: Shape configuration (`MlaConfigV2`).
    """

    # ---- Config field aliases (mirror MhaPrefillV2) -----------------
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth
    """V / O head depth (`d_pv`). For DeepSeek-V3 MLA: 128. Identical
    semantics to `MhaPrefillV2.DEPTH`: `MhaMmaOp` is specialized on
    this via `config.mha()` so the V and PV machinery shares verbatim."""

    comptime NUM_HEADS = Self.config.num_heads
    comptime NUM_KV_HEADS = Self.config.num_kv_heads
    comptime NUM_WARPS = Self.config.num_warps

    # ---- MLA-specific aliases ---------------------------------------
    comptime D_QK = Self.config.d_qk
    """Q / K depth (`d_nope + d_rope`). For DeepSeek-V3 MLA: 192."""

    comptime D_NOPE = Self.config.depth
    """Alias for `DEPTH` exposing the nope-segment semantic name.
    `D_NOPE == DEPTH == d_pv`: DeepSeek-V3 MLA does not RoPE V."""

    comptime D_ROPE = Self.config.d_rope
    """RoPE-applied segment depth on Q and K. For DeepSeek-V3 MLA: 64."""

    comptime CACHE_DEPTH = Self.config.cache_depth
    """Latent K cache row width. For DeepSeek-V3 MLA: 576. The gap
    between `D_NOPE` (128) and `ROPE_CACHE_OFFSET` (512) is reserved
    / unused but counted in the cache stride. Matches the existing
    BF16 MLA path at `mla_prefill.mojo:54`."""

    comptime ROPE_CACHE_OFFSET = Self.config.rope_cache_offset
    """Column offset of `k_rope` within the latent cache row. For
    DeepSeek-V3 MLA: 512 (`k_nope` at `[:, :128]`, gap, `k_rope` at
    `[:, 512:576]`)."""

    # ---- Derived block-level shape constants ------------------------
    comptime NUM_THREADS = Self.NUM_WARPS * 64

    # MhaMmaOp specialization on `config.mha()` — the MFMA shape, SMEM
    # sub-block geometry, K loader, V loader, PV path, and exp2 helper
    # all live here and are shared verbatim with MhaPrefillV2 at
    # `d_pv = depth`. `config.mha()` is a comptime field copy — no
    # runtime cost.
    comptime _MmaOp = MlaMmaOp[Self.config.dtype, Self.config.mha()]

    # Q/K K-dim tile counts. Q spans `_NUM_Q_K_TILES` MFMA base tiles
    # for the full `d_qk`; the nope segment of K spans
    # `_NUM_K_NOPE_TILES`, the rope segment spans `_NUM_K_ROPE_TILES`.
    # By construction `_NUM_K_NOPE_TILES + _NUM_K_ROPE_TILES ==
    # _NUM_Q_K_TILES` (linear-in-K accumulation across the two
    # segments into one `att` accumulator).
    comptime _NUM_Q_K_TILES = Self.D_QK // Self._MmaOp.MMA_K
    """`d_qk / MMA_K`. For FP8 (MMA_K=64) at d_qk=192: 3. For BF16
    (MMA_K=16) at d_qk=192: 12."""

    comptime _NUM_K_NOPE_TILES = Self.DEPTH // Self._MmaOp.MMA_K
    """`d_nope / MMA_K`. For FP8 at d_nope=128: 2. For BF16 at
    d_nope=128: 8."""

    comptime _NUM_K_ROPE_TILES = Self.D_ROPE // Self._MmaOp.MMA_K
    """`d_rope / MMA_K`. For FP8 at d_rope=64: 1. For BF16 at
    d_rope=64: 4."""

    # ---- FP8 KV>=128 32x32x64 shape predicate ----------------------
    # The FP8 / KV>=128 / 32x32x64 shape predicate: FP8 +
    # KV_BLOCK >= 128 + 32x32x64 MMA (not the 16x16x128 path). Both the
    # `_CADENCE` streamed-K gate and the `_FP32_SOFTMAX_SCORES`
    # in-place-FP32 gate key off this; factored out so `_SOFTMAX_DTYPE`
    # (defined before the gates) can also consult it.
    comptime _FP8_KV128_32x64 = (
        Self.config.dtype.is_float8()
        and Self.KV_BLOCK >= 128
        and not Self._MmaOp.FP8_MMA_K_128
    )

    # ---- FP32-in-place-scores gate ---------------------------------
    # Run softmax end-to-end in FP32 on the QK MFMA output tile
    # (`att_block`), then `v_cvt_pk_fp8_f32 op_sel:[0,0,1]` collapses
    # FP32 scores 4:1 directly into the FP8 PV operand — no separate
    # FP16 `att_block`, no FP8 `p_block` widen (gfx950 has no FP16->FP8
    # pack; FP16 storage always pays the FP16->FP32 widen). The FP32
    # score tile (64 VGPR vs FP16's 32) is NOT extra budget here — it IS
    # the scores — and the streamed-K band (coalesced with V) already
    # freed the 96-VGPR persistent-K reservation that made a NAIVE FP32
    # flip spill.
    #
    # This is the production path for the FP8 / KV>=128 / 32x32x64 shape:
    # the QK MFMA produces FP32 directly (gfx950 forbids FP8xFP8->FP16,
    # so the FP16-softmax alternative does not even compile here), so the
    # gate is the shape predicate — on whenever `MlaPrefillV2` is the
    # routed kernel.
    comptime _FP32_SOFTMAX_SCORES = Self._FP8_KV128_32x64
    """In-place FP32 softmax gate. On for the FP8 / KV>=128 / 32x32x64
    shape (the shape `MlaPrefillV2` serves): `_SOFTMAX_DTYPE` is FP32,
    routing the FP32 in-place softmax + direct FP32->FP8 cast path."""

    # ---- Softmax dtype and Q prescale dispatch (mirror MhaPrefillV2) -
    # Conditional FP16 narrowing for the FP8 KV>=128 path. At KV=128, the
    # att_block live set dominates VGPR pressure; halving it (FP32→FP16,
    # 16→8 VGPR/lane) lets the compiler pack the cast `v_cvt_pk_fp8_f32`
    # chain into fewer VGPRs, easing the v0 funnel pressure.
    # The single-buffer collapse (no att_block_0/1 ping-pong) is
    # preserved — the QK consumer in `MlaPrefillV2._attend_exact`
    # handles the FP16-target branch by holding the FP32 partials in
    # per-(n,m) SIMD locals only (no kernel-scope FP32 buffer), narrowing
    # to FP16 at write-back.
    #
    # When the in-place-FP32 sub-gate (`_FP32_SOFTMAX_SCORES`) is on, this
    # stays FP32 even at FP8 KV>=128 — the QK MFMA output IS the softmax
    # workspace, telescoped 4:1 to FP8 at the PV boundary. The
    # FP16-narrowing branch below is the DEFAULT (sub-gate off).
    comptime _SOFTMAX_DTYPE = (
        DType.float16 if Self.config.dtype.is_float8()
        and Self.KV_BLOCK >= 128
        and not Self._FP32_SOFTMAX_SCORES else DType.float32
    )
    """Softmax accumulator dtype. FP32 for BF16 and FP8 KV<128; FP16
    for FP8 KV>=128 (gfx950 has no packed BF16 element-wise; FP16 is
    the right narrowing); FP32 again at FP8 KV>=128 when
    `_FP32_SOFTMAX_SCORES` (in-place FP32)."""

    comptime prescale_q = not Self.config.dtype.is_float8()
    """Q prescale at load time. True for BF16 (single FP32 multiply
    per Q element at load); False for FP8 (avoids FP32→FP8 precision
    loss; post-QK scale is applied on `att_block` instead)."""

    # ---- Streamed-K-band gate --------------------------------------
    # Gates the "stream K from SMEM inside the QK consumer cluster" path.
    # When True (and the dtype/shape match), the main-loop C0/C4 QK
    # consumer loads K from `k_smem[slot]` into a FUNCTION-LOCAL band
    # right at the top of the cluster, instead of consuming the
    # kernel-scope `k_nope_reg`/`k_rope_reg` that the persistent path
    # preloads in C3/C7. The function-local band dies at the QK helper's
    # return, so the register allocator can pack it into the same
    # physical VGPRs that V (also cluster-local) uses later in the
    # iteration — freeing the 96-VGPR kernel-scope K reservation.
    #
    # Gate condition: FP8 + KV_BLOCK >= 128 + 32x32x64 MMA (not the
    # 16x16x128 path). This is the shape where the kernel spills without
    # the band-sharing and where the K/V band-sharing buys budget.
    comptime _CADENCE = Self._FP8_KV128_32x64
    """Streamed-K-band gate. On for the FP8 / KV>=128 / 32x32x64 shape
    (the shape `MlaPrefillV2` serves). No effect on other shapes
    (BF16, KV=64, FP8 16x16x128): the comptime condition gates it off
    there."""

    # ---- Register-tile layouts -------------------------------------
    # Q_LAYOUT_MLA mirrors MhaMmaOp.Q_LAYOUT but with D_QK as the K-dim
    # extent instead of DEPTH. The full Q tile holds both q_nope and
    # q_rope contiguously: indices `[0, _NUM_K_NOPE_TILES)` cover
    # q_nope, indices `[_NUM_K_NOPE_TILES, _NUM_Q_K_TILES)` cover
    # q_rope. The two-segment QK slices Q via `.tile[]()` — pure re-tag,
    # no data motion.
    comptime Q_LAYOUT_MLA = row_major[
        Self.Q_BLOCK_SIZE // Self._MmaOp.MMA_M,
        Self.D_QK // Self._MmaOp.MMA_K,
        (Self._MmaOp.MMA_M * Self._MmaOp.MMA_K) // 64,
    ]()
    """Q register tile at d_qk. For FP8 KV_BLOCK=64 d_qk=192: shape
    `[1, 3, 32]` (1 row base × 3 col base × 32 FP8 elts/lane). For
    BF16: `[1, 12, 8]`."""

    # K_NOPE_LAYOUT === MhaMmaOp.K_LAYOUT — the MhaMmaOp is bound to
    # `config.mha()` which carries `depth = d_nope`, so K_LAYOUT
    # already encodes the nope geometry. Exposed here as an alias for
    # symmetry with K_ROPE_LAYOUT.
    comptime K_NOPE_LAYOUT = Self._MmaOp.K_LAYOUT
    """K_nope register tile, identical to the MHA path's K register tile
    at d=d_nope. Re-exposed from `_MmaOp.K_LAYOUT`."""

    # K_ROPE_LAYOUT sibling at d_rope. For FP8 at d_rope=64 MMA_K=64
    # the col base-tile count collapses to 1 → `[2, 1, 32]`. For BF16
    # at d_rope=64 MMA_K=16: `[2, 4, 8]`.
    comptime K_ROPE_LAYOUT = row_major[
        Self.KV_BLOCK // Self._MmaOp.MMA_M,
        Self.D_ROPE // Self._MmaOp.MMA_K,
        (Self._MmaOp.MMA_M * Self._MmaOp.MMA_K) // 64,
    ]()
    """K_rope register tile at d_rope. Loaded by `MhaMmaOp.load_K`
    verbatim; col loop = width 1 (FP8) / 4 (BF16). SMEM source
    `smem_layout_k_rope`."""

    # ---- Latent-cache stride for gmem TileTensor layouts -----------
    # MLA's latent cache row width is `CACHE_DEPTH` (= 576 for
    # DeepSeek-V3) — NOT `DEPTH`. Row stride per kv-head is therefore
    # `NUM_KV_HEADS * CACHE_DEPTH`. k_nope and k_rope tiles both
    # navigate this stride; they differ only in column extent (`D_NOPE`
    # vs `D_ROPE`) and starting column (`0` vs `ROPE_CACHE_OFFSET`).
    comptime _KV_ROW_STRIDE_MLA = Self.NUM_KV_HEADS * Self.CACHE_DEPTH
    """Per-row gmem stride for the latent K cache. For DeepSeek-V3
    with NUM_KV_HEADS=1: 576 elements."""

    comptime _Q_ROW_STRIDE = Self.NUM_HEADS * Self.D_QK
    """Per-row gmem stride for Q. For DeepSeek-V3 MLA, Q is stored as
    `q_nope ∥ q_rope` row-major at `d_qk` per head."""

    # ---- K SMEM swizzle (mirror MhaPrefillV2) ----------------------
    # Two-XOR swizzle composes to byte-level `bit4 ^= bit8; bit5 ^=
    # bit9`. Bank-conflict-free on the 32-lanes-per-row MFMA access.
    # Byte-positional, so it works identically for k_nope (KV_BLOCK x
    # D_NOPE) and k_rope (KV_BLOCK x D_ROPE) sub-blocks — the swizzle
    # operates on per-sub-block byte offsets where `K_SUB_COLS_BYTES`
    # is dtype-determined and the same for both segments. See
    # MhaPrefillV2 for the conflict-residual derivation.
    comptime k_swizzle = Optional(Swizzle(1, 0, 4))
    comptime k_swizzle2 = Optional(Swizzle(1, 1, 4))

    # Shared K loader for both k_nope and k_rope DMAs. Byte-positional
    # swizzle is invariant to the segment width, so a single
    # specialization serves both.
    comptime KTileLoader = SubTileLoaderLDS[
        Self.config.dtype, Self.k_swizzle, Self.k_swizzle2
    ]

    # V cooperative DMA: 8-row sub-tile loader matching MhaPrefillV2.
    # V at MLA is identical to V at MHA — d_pv = depth = 128. No
    # rope on V (DeepSeek-V3 MLA decision).
    comptime VTileLoader = SubTileLoaderLDS_st_8x32[
        Self.config.dtype,
        Self.KV_BLOCK,
        Self.DEPTH,
        64 if Self.config.dtype.is_float8() else 32,
        Self.NUM_THREADS,
    ]

    # ---- Per-tile gmem layout types --------------------------------
    # Single-base K tile: `(KV_BLOCK, CACHE_DEPTH)` with latent-cache
    # row stride. Spans the full latent cache row — nope cols
    # `[0, D_NOPE)` + reserved gap + rope cols `[ROPE_CACHE_OFFSET,
    # ROPE_CACHE_OFFSET + D_ROPE)`. Built once per K[t] via
    # `_make_k_full_tile`; consumed by `_MlaKDmaPair` which provides
    # nope / rope sub-tile views to a single shared `KTileLoader`.
    # Mirrors the reference's single LDS-base pattern for K (`v226`) on
    # the gmem side: one SGPR buffer resource covers both segments. The
    # gap between `D_NOPE` and `ROPE_CACHE_OFFSET` is never read; the
    # bounding `_get_bounds(tile)` is `(KV_BLOCK - 1) *
    # _KV_ROW_STRIDE_MLA + CACHE_DEPTH` bytes, comfortably under the
    # 32-bit `num_records` cap.
    comptime _KFullPerTileLayoutT = TileLayout[
        Coord[
            ComptimeInt[Self.KV_BLOCK], ComptimeInt[Self.CACHE_DEPTH]
        ].element_types,
        Coord[
            ComptimeInt[Self._KV_ROW_STRIDE_MLA], ComptimeInt[1]
        ].element_types,
    ]

    # V tile: (KV_BLOCK, DEPTH) with latent-cache row stride. V lives
    # at columns `[0, D_NOPE)` of the latent cache — the nope segment
    # is reinterpreted as V at the consumer (PV matmul). The MHAOperand
    # passed for V should anchor at `head_dim_idx=0`.
    comptime _VPerTileLayoutT = TileLayout[
        Coord[
            ComptimeInt[Self.KV_BLOCK], ComptimeInt[Self.DEPTH]
        ].element_types,
        Coord[
            ComptimeInt[Self._KV_ROW_STRIDE_MLA], ComptimeInt[1]
        ].element_types,
    ]

    # ---- K SMEM slot byte geometry ---------------------------------
    # k_nope slot: KV_BLOCK rows × D_NOPE cols, linearized into
    # `KV_BLOCK × NUM_BLOCK_COLS_K_NOPE` rows × K_SUB_COLS cols.
    # k_rope slot: same form with NUM_BLOCK_COLS_K_ROPE = D_ROPE /
    # K_SUB_COLS. For FP8 at d_rope=64, K_SUB_COLS=64 → 1 column
    # sub-block → k_rope slot is half the rows of k_nope slot.
    comptime _NUM_BLOCK_COLS_K_NOPE = Self.D_NOPE // Self._MmaOp.K_SUB_COLS
    """`D_NOPE / K_SUB_COLS`. For FP8 at d=128: 2. For BF16 at d=128: 4."""

    comptime _NUM_BLOCK_COLS_K_ROPE = Self.D_ROPE // Self._MmaOp.K_SUB_COLS
    """`D_ROPE / K_SUB_COLS`. For FP8 at d_rope=64: 1 (single col
    sub-block). For BF16 at d_rope=64: 2."""

    comptime _NUM_BLOCK_COLS_K = (
        Self._NUM_BLOCK_COLS_K_NOPE + Self._NUM_BLOCK_COLS_K_ROPE
    )
    """K-coresident slab: total col sub-blocks across the unified
    K row. For FP8 KV=64: 2 + 1 = 3. For BF16 KV=64: 8 + 4 = 12."""

    comptime _K_NOPE_SLOT_ROWS = (Self.KV_BLOCK * Self._NUM_BLOCK_COLS_K_NOPE)
    """Linearized row count of one k_nope SMEM slot. For FP8 at
    KV_BLOCK=64 D_NOPE=128: 128 (= 64 × 2)."""

    comptime _K_ROPE_SLOT_ROWS = (Self.KV_BLOCK * Self._NUM_BLOCK_COLS_K_ROPE)
    """Linearized row count of one k_rope SMEM slot. For FP8 at
    KV_BLOCK=64 D_ROPE=64: 64 (= 64 × 1)."""

    comptime _K_SLOT_ROWS = Self.KV_BLOCK * Self._NUM_BLOCK_COLS_K
    """K-coresident slab: linearized row count of the unified K
    SMEM slot covering both nope and rope. For FP8 KV=64: 192 (=
    64 × 3). The nope sub-view occupies rows `[0, _K_NOPE_SLOT_ROWS)`;
    the rope sub-view occupies rows
    `[_K_NOPE_SLOT_ROWS, _K_NOPE_SLOT_ROWS + _K_ROPE_SLOT_ROWS)`."""

    comptime _K_ROPE_SUBBLOCK_OFFSET = (
        (Self.KV_BLOCK // Self._MmaOp.K_SUB_ROWS) * Self._NUM_BLOCK_COLS_K_NOPE
    )
    """K-coresident slab: sub-block ID offset for the rope half
    of the unified slot. Equals `_K_NOPE_SLOT_ROWS / K_SUB_ROWS`. For
    FP8 KV=64 (K_SUB_ROWS=32): 4. The DMA helper for k_rope adds this
    offset to its locally-computed `subblock_id` so the
    `tile[K_SUB_ROWS, K_SUB_COLS](id, 0)` lands in the rope rows of
    the unified slot."""

    comptime _K_ROPE_TILE_IDX = Self._K_NOPE_SLOT_ROWS // Self._K_ROPE_SLOT_ROWS
    """K-coresident slab: tile index for `.tile[_K_ROPE_SLOT_ROWS,
    K_SUB_COLS](_K_ROPE_TILE_IDX, 0)` to view the rope sub-region of
    the unified slot. Equals `D_NOPE / D_ROPE`. For DeepSeek-V3 MLA:
    2 (nope is 2x the depth of rope). The struct-level integer
    division above is exact iff `_K_NOPE_SLOT_ROWS` is a multiple of
    `_K_ROPE_SLOT_ROWS`: true for DeepSeek-V3 MLA where the col
    sub-block counts share `K_SUB_COLS`."""

    comptime _NUM_BLOCK_COLS_V = Self.DEPTH // Self._MmaOp.V_SUB_COLS
    comptime _V_SLOT_ROWS = Self.KV_BLOCK * Self._NUM_BLOCK_COLS_V

    # ---- Per-tile gmem tile constructors ---------------------------
    @staticmethod
    @always_inline
    def _make_k_full_tile[
        k_t: MHAOperand,
        //,
    ](
        k_op: k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
    ) -> TileTensor[
        Self.config.dtype, Self._KFullPerTileLayoutT, ImmutAnyOrigin
    ]:
        """Builds a single per-tile gmem TileTensor spanning the full
        CACHE_DEPTH columns at `head_dim_idx=0`.

        Used by `_MlaKDmaPair` to create one `KTileLoader` serving both
        nope (cols `[0, D_NOPE)`) and rope (cols `[ROPE_CACHE_OFFSET,
        ROPE_CACHE_OFFSET + D_ROPE)`) via sub-tile views. The DMA never
        touches the reserved gap between `D_NOPE` and
        `ROPE_CACHE_OFFSET`; the `KTileLoader` only issues
        `buffer_load_*_lds` for the active sub-views passed to
        `.load()`.

        Anchored at `head_dim_idx=0`. In production both `k_nope_op`
        and `k_rope_op` are views over the SAME latent cache backing
        buffer; the rope offset is captured here in the sub-tile column
        index rather than via per-operand `head_dim_offset`. Callers
        therefore pass `k_nope_op` (which anchors at column 0).
        """
        comptime assert (
            k_t.dtype == Self.config.dtype
        ), "MlaPrefillV2Core: k op dtype must equal `config.dtype`"
        return rebind[
            TileTensor[
                Self.config.dtype, Self._KFullPerTileLayoutT, ImmutAnyOrigin
            ]
        ](
            k_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._KFullPerTileLayoutT(
                    Coord(Idx[Self.KV_BLOCK], Idx[Self.CACHE_DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE_MLA], Idx[1]),
                ),
                head_dim_idx=UInt32(0),
            )
        )

    @staticmethod
    @always_inline
    def _make_v_tile[
        v_t: MHAOperand,
        //,
    ](
        v_op: v_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
    ) -> TileTensor[
        Self.config.dtype, Self._VPerTileLayoutT, ImmutAnyOrigin
    ]:
        """Builds the per-tile V gmem TileTensor. V at MLA is the nope
        segment of the latent cache row (columns `[0, D_NOPE)`): same
        anchor as the nope cols of `_make_k_full_tile` but consumed for the
        PV matmul instead of the QK matmul. Row stride matches the latent
        cache.
        """
        comptime assert (
            v_t.dtype == Self.config.dtype
        ), "MlaPrefillV2Core: V dtype must equal `config.dtype`"
        return rebind[
            TileTensor[Self.config.dtype, Self._VPerTileLayoutT, ImmutAnyOrigin]
        ](
            v_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._VPerTileLayoutT(
                    Coord(Idx[Self.KV_BLOCK], Idx[Self.DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE_MLA], Idx[1]),
                ),
                head_dim_idx=UInt32(0),
            )
        )

    # ---- Cooperative DMA producers ---------------------------------
    # Both K segments DMA through a single `KTileLoader` built from the
    # unified `(KV_BLOCK, CACHE_DEPTH)` gmem tile — one SGPR buffer
    # resource per K[t] instead of two. See `_MlaKDmaPair` for the
    # per-segment warp partition + sub-tile addressing.

    # ---- Per-tile layout types for Q (d_qk wide) --------------------
    comptime _QPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.D_QK]].element_types,
        Coord[ComptimeInt[Self._Q_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]

    # Per-(batch, head) 2D layout for the output. O is at d_pv (same as
    # the MHA path) because V is not RoPE'd.
    comptime _OPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[
            ComptimeInt[Self.NUM_HEADS * Self.DEPTH], ComptimeInt[1]
        ].element_types,
    ]

    # Layout type aliases on the `_MmaOp` (`MhaMmaOp`) member — used as
    # `type_of(...)` in RegTile signatures for V, O, ATT, etc.
    comptime _Q_LAYOUT_MLA_T = type_of(Self.Q_LAYOUT_MLA)
    comptime _ATT_LAYOUT_T = type_of(Self._MmaOp.ATT_LAYOUT)
    comptime _ATT_BF16_FULL_LAYOUT_T = type_of(Self._MmaOp.ATT_BF16_FULL_LAYOUT)
    comptime _O_LAYOUT_T = type_of(Self._MmaOp.O_LAYOUT)
    comptime _O_T_LAYOUT_T = type_of(Self._MmaOp.O_T_LAYOUT)

    # ---- V DMA helper -------------------------------------------------
    @staticmethod
    @always_inline
    def _dma_v[
        v_t: MHAOperand,
        //,
        v_full_v227: Bool = False,
    ](
        v_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        v_op: v_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        w_id: Int,
        l_id: Int,
    ):
        """Issues the V[t] DMA into `v_smem_slot` from the latent cache
        (head_dim columns `[0, D_NOPE)`, the nope segment which is V).
        V uses the same identity-swizzle cooperative loader as the MHA
        path; DeepSeek-V3 MLA does not RoPE V.

        Parameters:
            v_full_v227: `v227` V adapter WRITE (Bool).
                Default False → byte-identical contiguous fill (the
                production MLA path builds the exact same `VTileLoader`).
                When True, the producer reorganizes the cooperative DMA
                into the reference's chunk-stepped LDS layout (the `W` of
                the adapter `W∘R` pair) so the consumer's `v227` transpose
                read (`precompute_v_lane_base[v_full_v227=True]` +
                `load_V_frag[v_full_v227=True]`) reads the standard PV
                fragment bank-conflict-free. The default-on V LDS adapter
                for `MlaPrefillV2`; production MLA passes
                `v_full_v227=False`. `-D v_full_v227`.
        """
        var v_gmem_tile = Self._make_v_tile(v_op, batch_idx, kv_head_idx, t)
        # Build the loader with the requested adapter layout. For the
        # default (v_full_v227=False) this is the byte-identical
        # `Self.VTileLoader` type.
        var v_loader = SubTileLoaderLDS_st_8x32[
            Self.config.dtype,
            Self.KV_BLOCK,
            Self.DEPTH,
            64 if Self.config.dtype.is_float8() else 32,
            Self.NUM_THREADS,
            v_full_v227=v_full_v227,
        ](v_gmem_tile)
        # Inline pass-through: loader is built from `v_gmem_tile` and
        # the same tile is the src, so `scalar_offset` evaluates to 0.
        v_loader.load(
            v_smem_slot,
            v_gmem_tile,
            w_id,
            l_id,
            scalar_offset=Int(v_gmem_tile.ptr) - v_loader.bc.get_base_ptr(),
        )

    # ---- Q load + scale (d_qk wide) -----------------------------------
    @staticmethod
    @always_inline
    def load_q[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[
            Self.config.dtype, layout, Engine=DefaultEngine[], ...
        ],
    ) -> RegTile[Self.config.dtype, Self._Q_LAYOUT_MLA_T, MutUntrackedOrigin]:
        """Loads the warp's Q sub-tile at d_qk from gmem into the row_l
        register tile. Mirrors `MhaPrefillV2.load_q` but iterates
        `_NUM_Q_K_TILES = D_QK // MMA_K` K-dim base tiles instead of
        `DEPTH // MMA_K`.

        For FP8 at d_qk=192 (MMA_K=64) the loop runs 3 times: two
        iterations cover the nope half (col 0..127) and one covers the
        rope half (col 128..191). Q is stored in gmem as `q_nope ∥
        q_rope` contiguously, so the underlying buffer_load_lds reads
        are uniform across the 3 tiles: no nope/rope branching here.

        Parameters:
            layout: `TensorLayout` of the `q_warp_2d` gmem tile.

        Args:
            q_warp_2d: The warp's Q sub-tile in gmem at `d_qk` width,
                storing `q_nope` and `q_rope` contiguously.
        """
        comptime _BK = Self._MmaOp.MMA_K
        comptime _num_k_tiles = Self.D_QK // _BK
        comptime _q_thread_layout = col_major[
            Self.Q_BLOCK_SIZE, WARP_SIZE // Self.Q_BLOCK_SIZE
        ]()

        var q_reg = reg_alloc[Self.config.dtype](Self.Q_LAYOUT_MLA)
        var q_loader = RegTileLoader[
            Self.config.dtype, _q_thread_layout, warp_scope=True
        ](q_warp_2d)

        comptime if Self.config.dtype == .float8_e4m3fn:
            # FP8: per-lane fragment = 32 FP8 = 32 B. To match the MFMA's
            # B-operand lane layout (which is the same convention as the
            # A-operand K loader in `MhaMmaOp.load_K` FP8 32x32x64 path),
            # lane `lid` must own 32 *contiguous* K-cols in its per-lane
            # fragment: `row = lid % 32`, `col_base = (lid // 32) * 32`.
            # The previous `RegTileLoader` + `col_major[32, 2]` distribute
            # path on a `[32, 32]` half-tile produced a *striped*
            # fragment (lane 0: cols {0..15, 32..47}) that mismatched K
            # and led to wrong MFMA pairings — root cause of the MLA
            # FP8 vs flare cos_sim drop. See `MhaPrefillV2.load_q` for
            # the matching fix on the MHA path.
            comptime _row_stride = type_of(q_warp_2d).static_stride[0]
            var lid = Int(lane_id())
            var row_offset = lid % 32
            var col_base = (lid // 32) * 32

            var bc = make_amd_buffer_resource(q_warp_2d)
            var q_reg_v = q_reg.vectorize[1, 1, 16]()
            comptime for j in range(_num_k_tiles):
                var base_off_lo = Int32(
                    row_offset * _row_stride + j * _BK + col_base
                )
                var base_off_hi = base_off_lo + Int32(16)
                var lo = bc.load[Self.config.dtype, 16](base_off_lo)
                var hi = bc.load[Self.config.dtype, 16](base_off_hi)
                q_reg_v[0, j, 0] = rebind[type_of(q_reg_v[0, j, 0])](lo)
                q_reg_v[0, j, 1] = rebind[type_of(q_reg_v[0, j, 1])](hi)
        else:
            # BF16: per-lane fragment = 8 BF16 = 16 B, one buffer_load
            # per K-tile.
            comptime for j in range(_num_k_tiles):
                var src = q_warp_2d.tile[Self.Q_BLOCK_SIZE, _BK](0, j)
                var dst = q_reg.tile[1, 1, 8](0, j, 0).reshape(
                    row_major[1, 8]()
                )
                q_loader.load(dst, src.vectorize[1, 8]())

        return q_reg

    @staticmethod
    @always_inline
    def _load_q_and_scale_mla[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[
            Self.config.dtype, layout, Engine=DefaultEngine[], ...
        ],
        scale_log2e: Float32,
    ) -> RegTile[Self.config.dtype, Self._Q_LAYOUT_MLA_T, MutUntrackedOrigin]:
        """Loads Q (d_qk wide) and (when `Self.prescale_q` is True)
        prescales it by `scale * log2e`.

        Same prescale logic as `MhaPrefillV2._load_q_and_scale`. For
        BF16 the prescale multiplies in FP32 per fragment then casts
        back; for FP8 it's comptime-elided (post-QK scale lands on
        `att_block` instead).

        The prescale applies uniformly to BOTH q_nope and q_rope
        fragments since Q is stored as `q_nope ∥ q_rope` at the
        operand boundary; there is no need for a nope/rope-specific
        scale, and matmul-linear-in-K means the two-segment QK still
        sums to the correct prescaled QK^T."""
        var q_reg = Self.load_q(q_warp_2d)

        comptime if Self.prescale_q:
            comptime _H = Self._Q_LAYOUT_MLA_T.static_shape[0]
            comptime _W = Self._Q_LAYOUT_MLA_T.static_shape[1]
            comptime _F = Self._Q_LAYOUT_MLA_T.static_shape[2]
            var q_v = q_reg.vectorize[1, 1, _F]()
            comptime assert q_v.flat_rank == 3
            comptime for h in range(_H):
                comptime for w in range(_W):
                    q_v[h, w, 0] = (
                        q_v[h, w, 0].cast[.float32]() * scale_log2e
                    ).cast[Self.config.dtype]()

        return q_reg

    # ---- PV cluster bodies (delegate to MhaPrefillV2-shaped helpers) -
    #
    # V/O/ATT machinery is unchanged from the MHA path: same MMA shape,
    # same PV-A layout, same softmax recurrence (online softmax is dtype-
    # generic on `T_att`). We define the strip and whole PV bodies
    # locally so cluster fns can call them with MlaPrefillV2Core-scoped
    # type signatures; the bodies forward to the same `MhaMmaOp.mma_PV`
    # and `OnlineSoftmax` calls as the MHA path's.

    comptime _ATT_PER_LANE = (Self.KV_BLOCK * Self.Q_BLOCK_SIZE) // 64
    """Per-lane element count for the FP32 att tile."""

    comptime _PV_A_FRAG = 32 if Self.config.dtype.is_float8() else 8
    """Per-lane PV-A fragment width."""

    comptime _MMA_K = 64 if Self.config.dtype.is_float8() else 16
    """Per-MFMA K-dim."""

    comptime _NUM_PV_SUBTILES = Self.KV_BLOCK // Self._MMA_K
    """Number of MMA_K-row PV strips in one V tile."""

    @staticmethod
    @always_inline
    def _att_bf16_full(
        mut dst: RegTile[
            Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
        ],
        att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
    ):
        """Bulk-casts `att_block` (FP32) to the PV-A input dtype, writing
        into the caller-provided persistent destination `dst`.

        The persistent destination is required so `att_block` can be
        freed for the next QK fill in the same cluster, breaking the
        att_block_0/1 ping-pong dependency.

        Same shape as `MhaPrefillV2._att_bf16_full` but always casts
        from FP32 (`_SOFTMAX_DTYPE` is float32 under `_FP32_SOFTMAX_SCORES`).
        For BF16 dst: per-strip cast + half-slice into `_NUM_PV_SUBTILES`
        sub-tiles. For FP8 dst: pairwise `v_cvt_pk_fp8_f32` + join,
        producing 32 FP8/lane per sub-tile.

        Cast safety: P ∈ [0, 1] post-softmax is provably bounded and
        non-NaN, so the generic `SIMD.cast` wrapper (v_med3_f32 clamp
        ±448 + v_cmp_u_f32 NaN-scrub + v_cndmask) is unnecessary work.
        Bare `v_cvt_pk_fp8_f32` mirrors `mla_prefill`'s
        `PRegisterBuffer.mma_tile` lazy-cast safety analysis.
        """
        var src_v = att_block.vectorize[1, 1, 16]()
        comptime if Self.config.dtype == .float8_e4m3fn:
            # FP8 path: 2 source strips (16 FP32/lane each) → 1 sub-tile
            # (32 FP8/lane), `_NUM_PV_SUBTILES=1` at MLA-FP8 KV=64,
            # or 2 sub-tiles at KV=128.
            #
            # IN-PLACE-SAFE: `dst` MAY ALIAS `att_block`'s OWN low
            # quarter, and the per-sub-tile read-then-store ordering
            # keeps the aliased store safe. See `_qk_collapse_inplace` for
            # the full argument.
            var dst_v = dst.vectorize[1, 1, 32]()
            comptime for sub in range(Self._NUM_PV_SUBTILES):
                var fp32_lo = src_v[2 * sub, 0, 0].cast[.float32]()
                var fp32_hi = src_v[2 * sub + 1, 0, 0].cast[.float32]()
                var fp8_lo = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_lo)
                var fp8_hi = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_hi)
                dst_v[sub, 0, 0] = fp8_lo.join(fp8_hi)
        else:
            # BF16 path: src is FP32. Each source strip (16 FP32/lane)
            # casts to 16 BF16 then half-slices into 2 sub-tiles.
            # `_NUM_PV_SUBTILES = KV_BLOCK // MMA_K`.
            var dst_v = dst.vectorize[1, 1, 8]()
            comptime for sub in range(Self._NUM_PV_SUBTILES):
                comptime _strip, _half = divmod(sub, 2)
                var bf16 = src_v[_strip, 0, 0].cast[Self.config.dtype]()
                dst_v[sub, 0, 0] = bf16.slice[8, offset=_half * 8]()

    # ---- Sequential FP32 cadence body ------------------------------
    @staticmethod
    @always_inline
    def _qk_collapse_inplace(
        mut att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
    ) -> RegTile[
        Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
    ]:
        """Collapse the post-softmax FP32 score tile 4:1 to FP8 IN PLACE,
        into `att_block`'s OWN low quarter: the `v_cvt_pk_fp8_f32`
        / `op_sel:[0,0,1]` telescope. Returns an FP8 view that
        ALIASES `att_block`'s storage (its low `_ATT_BF16_FULL_LAYOUT`
        worth of dwords = the FP8 P), so NO separate FP8 P tile is
        allocated and the FP32 score tile and the FP8 P never coexist as
        distinct live ranges.

        The view is a DISTINCT `var` binding over the same pointer as
        `att_block` (`bitcast` to the FP8 scalar type), built with the
        same view-over-`.ptr` idiom as `o_normalized_view` in
        `MlaPrefillV2._attend_exact`. The borrow checker tracks
        bindings, not
        physical storage, so `p_view` and `att_block` are separate
        bindings: `_att_bf16_full(p_view, att_block)` is NOT a
        self-aliased mut+read.

        Why the aliased store is safe: `_att_bf16_full`'s FP8 branch reads
        each FP32 source strip into a SIMD value register (`fp32_lo`/
        `fp32_hi`) BEFORE the (now-aliasing) store of that sub-tile, and
        the per-sub-tile dest dwords never overlap a source dword that a
        LATER sub-tile still reads (the cast is layout-monotone; sub `i`
        writes the low `[8i, 8i+8)` dwords while consuming the `[32i,
        32i+32)` source dwords). This mirrors the reference's
        read-v60,v61-then-write-v60 ordering.

        Aliasing the destination over `att_block.ptr` (rather than a
        persistent caller-owned tile) is also the remat-avoidance lever:
        the cast cluster has nowhere to rematerialize a fresh score
        copy."""
        var p_view = RegTile[
            Self.config.dtype,
            Self._ATT_BF16_FULL_LAYOUT_T,
            MutUntrackedOrigin,
        ](
            att_block.ptr.bitcast[Scalar[Self.config.dtype]](),
            Self._MmaOp.ATT_BF16_FULL_LAYOUT,
        )
        Self._att_bf16_full(p_view, att_block)
        return p_view

    # ---- Output store ------------------------------------------------
    @staticmethod
    @always_inline
    def _store_o_to_gmem[
        output_dtype: DType,
        epilogue_chunk_width: Int = 1,
    ](
        o_reg_t: RegTile[.float32, Self._O_T_LAYOUT_T, MutUntrackedOrigin],
        epilogue_writer: RegTileEpilogue[output_dtype, epilogue_chunk_width],
        l_id: Int,
        valid_q_rows_in_warp: Int,
    ):
        """Writes the FP32 row_l rt_32x32 accumulator to gmem at d_pv.

        Identical to `MhaPrefillV2._store_o_to_gmem` because V/O are
        unchanged at MLA (DeepSeek-V3 does not RoPE V). The output
        depth is `DEPTH = d_pv`, same as the MHA path.
        """
        var q_in_tile = l_id & 31
        var d_extra = 4 if l_id >= 32 else 0
        var q_in_bounds = q_in_tile < valid_q_rows_in_warp

        comptime _D_FRAG = (Self.DEPTH * Self.Q_BLOCK_SIZE) // 64
        comptime for k_local in range(_D_FRAG):
            comptime i = k_local // 16
            comptime k_in_base = k_local % 16
            comptime d_within_4 = (k_in_base // 4) * 8 + (k_in_base % 4)
            var output_col = i * 32 + d_within_4 + d_extra
            var v_fp32 = Float32(o_reg_t.ptr[k_local])
            if q_in_bounds:
                comptime if output_dtype == .float32:
                    epilogue_writer.store(
                        rebind[SIMD[output_dtype, 1]](v_fp32),
                        m=q_in_tile,
                        n=output_col,
                    )
                else:
                    epilogue_writer.store(
                        v_fp32.cast[output_dtype](),
                        m=q_in_tile,
                        n=output_col,
                    )


# ===----------------------------------------------------------------------=== #
# _MlaKDmaPair — single-base K loader, mirrors the reference's v226 pattern
# ===----------------------------------------------------------------------=== #


struct _MlaKDmaPair[
    k_t: MHAOperand,
    //,
    config: MlaConfigV2,
](TrivialRegisterPassable):
    """One buffer-resource pair for both K-segments: mirrors the
    reference's single-base K pattern.

    A single `KTileLoader` instance bound to the full latent cache row
    `(KV_BLOCK, CACHE_DEPTH)` serves both segments. Both `dma_nope` and
    `dma_rope` reuse the same loader's SGPR buffer resource (one SRD
    per K[t] instead of two); they differ only in the scalar column
    offset of the source sub-tile passed to `.load()`.

    Cluster geography (unified K-coresident slab):
    - nope sub-blocks land in rows `[0, _K_NOPE_SLOT_ROWS)` of the
      `k_smem_slot`, sourced from gmem cols `[0, D_NOPE)`.
    - rope sub-blocks land in rows `[_K_NOPE_SLOT_ROWS,
      _K_NOPE_SLOT_ROWS + _K_ROPE_SLOT_ROWS)` of the `k_smem_slot`,
      sourced from gmem cols `[ROPE_CACHE_OFFSET,
      ROPE_CACHE_OFFSET + D_ROPE)`.

    Reference (`aiter_full.s:66`): a single `v226 = 0xc300 + …`
    LDS pointer covers the full 192-byte K row. The LDS DESTINATION is
    unified (one slab per phase) and the gmem SOURCE is unified (one SRD
    per K[t]).

    Buffer-resource bounds: `(KV_BLOCK - 1) * _KV_ROW_STRIDE_MLA +
    CACHE_DEPTH` bytes: for DeepSeek-V3 FP8 with KV_BLOCK=128 that's
    ~73 KiB, well under the `num_records` 32-bit cap.
    """

    # Alias to the parent type for comptime constant access. The
    # parent is the source of truth for `KTileLoader`, layouts, and
    # all derived comptime arithmetic.
    comptime _P = MlaPrefillV2Core[Self.config]

    # KV-conditional buffer-offset hoist: when KV_BLOCK >= 128 the
    # K_nope + K_rope DMAs share one wide buffer resource and the
    # hoist collapses both call sites' `s_add base, partition_off;
    # s_sub base` chains into a single `s_add s_hoisted, imm`. At
    # KV_BLOCK = 64 the hoist costs +9 SGPRs pinning the hoisted base,
    # evicting registers from the softmax/MMA pipeline. Gated here so
    # KV=64 stays per-call (legacy codegen) while KV>=128 shares one
    # SGPR base across both sub-DMAs.
    comptime _HOIST = Self._P.KV_BLOCK >= 128

    @__allow_legacy_any_origin_fields
    var k_full_gmem_tile: TileTensor[
        Self.config.dtype, Self._P._KFullPerTileLayoutT, ImmutAnyOrigin
    ]
    """Full `(KV_BLOCK, CACHE_DEPTH)` gmem tile anchored at
    head_dim_idx=0. Holds the per-tile ptr + layout consumed by
    `KTileLoader`; the loader uses the layout's bounds to compute the
    SRD `num_records`."""

    var k_loader: Self._P.KTileLoader
    """One buffer resource for the full K row. Reused across
    `dma_nope` and `dma_rope`: `SubTileLoaderLDS.load()` takes an
    explicit `scalar_offset`, computed inline at KV=64 (per-call,
    legacy codegen) and hoisted once at KV>=128 (single SGPR base
    across both sub-DMAs)."""

    var src_base_offset: Int
    """Hoisted byte offset of `k_full_gmem_tile.ptr` relative to
    `k_loader.bc.get_base_ptr()`. Computed once in `__init__` and
    consumed by `dma_nope` / `dma_rope` when `_HOIST` is true. When
    `_HOIST` is false the field is still populated (1 SGPR) but the
    DMAs recompute the scalar_offset per call site, preserving the
    pre-hoist codegen at KV=64."""

    @always_inline
    def __init__(
        out self,
        k_op: Self.k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
    ):
        """Build the full-row gmem tile and the shared `KTileLoader`.

        In production both `k_nope_op` and `k_rope_op` are paged-cache
        views over the SAME backing buffer (verified at the kernel
        docstring); the rope offset is captured here in the sub-tile
        column index inside `dma_rope` rather than the operand's
        `head_dim_offset`. Callers pass `k_nope_op` (which anchors at
        column 0 of the latent cache).
        """
        self.k_full_gmem_tile = Self._P._make_k_full_tile(
            k_op, batch_idx, kv_head_idx, t
        )
        self.k_loader = Self._P.KTileLoader(self.k_full_gmem_tile)
        # Hoist once: `k_full_gmem_tile.ptr - k_loader.bc.base` is the
        # runtime piece of every per-call scalar_offset. Computing it
        # here lets `dma_nope`/`dma_rope` collapse their scalar_offset
        # additions into a single `s_add s_hoisted, imm` per call when
        # `_HOIST` is true. When `_HOIST` is false the field is unused
        # (the comptime branch picks the per-call recompute), so DCE
        # eliminates the store.
        self.src_base_offset = (
            Int(self.k_full_gmem_tile.ptr) - self.k_loader.bc.get_base_ptr()
        )

    @always_inline
    def dma_nope(
        self,
        k_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        w_id: Int,
        l_id: Int,
    ):
        """Writes nope cols `[0, D_NOPE)` of the latent cache row to
        LDS rows `[0, _K_NOPE_SLOT_ROWS)` of the unified `k_smem_slot`.

        The gmem source is sliced from the full-row tile, so the nope
        and rope DMAs share the SAME `self.k_loader.bc` buffer resource.
        """
        comptime _K_SUB_ROWS = Self._P._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._P._MmaOp.K_SUB_COLS
        comptime _num_kv_subblocks = (
            (Self._P.KV_BLOCK // _K_SUB_ROWS) * (Self._P.D_NOPE // _K_SUB_COLS)
        )
        comptime _warps_per_subblock = (Self._P.NUM_WARPS // _num_kv_subblocks)
        comptime _rows_per_warp = _K_SUB_ROWS // _warps_per_subblock
        comptime _num_block_cols_k = Self._P.D_NOPE // _K_SUB_COLS
        comptime assert (
            Self._P.NUM_WARPS == _num_kv_subblocks * _warps_per_subblock
        ), (
            "MlaPrefillV2Core k_nope DMA: NUM_WARPS must divide evenly"
            " into the k_nope sub-block grid"
        )

        var subblock_id, row_strip = divmod(w_id, _warps_per_subblock)
        var sub_row, sub_col = divmod(subblock_id, _num_block_cols_k)

        # nope sub-tile: cols `[0, D_NOPE)` of the full row → sub_col
        # ranges over `[0, D_NOPE / K_SUB_COLS)`. No column-offset
        # shift needed because nope starts at column 0.
        var k_nope_src = self.k_full_gmem_tile.tile[_K_SUB_ROWS, _K_SUB_COLS](
            sub_row, sub_col
        ).tile[_rows_per_warp, _K_SUB_COLS](row_strip, 0)
        # KV-conditional hoist: KV=64 keeps the legacy per-call codegen
        # (`scalar_offset = Int(src.ptr) - bc.base` materialized inline
        # at the call site); KV>=128 splits the same value into the
        # hoisted `self.src_base_offset` (shared with `dma_rope` via a
        # single SGPR carried by the struct) plus the per-call
        # `Int(src.ptr) - Int(k_full_gmem_tile.ptr)` warp partition,
        # and passes `hoist_scalar_offset=True` so `load()` consumes the
        # caller-supplied SGPR runtime piece + comptime partition imm
        # instead of recomputing `Int(src_partitions.ptr) - dram_base`
        # per inner-iter. Both branches compute the SAME numeric
        # scalar_offset; they differ in WHICH SSA value the AMDGPU
        # backend sees as the runtime piece — `src_partitions.ptr`
        # (rematerialized per iter, legacy) or `self.src_base_offset`
        # (hoisted once across DMAs).
        comptime if Self._HOIST:
            self.k_loader.load[hoist_scalar_offset=True](
                k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](subblock_id, 0).tile[
                    _rows_per_warp, _K_SUB_COLS
                ](row_strip, 0),
                k_nope_src,
                scalar_offset=self.src_base_offset
                + (Int(k_nope_src.ptr) - Int(self.k_full_gmem_tile.ptr)),
            )
        else:
            self.k_loader.load(
                k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](subblock_id, 0).tile[
                    _rows_per_warp, _K_SUB_COLS
                ](row_strip, 0),
                k_nope_src,
                scalar_offset=Int(k_nope_src.ptr)
                - self.k_loader.bc.get_base_ptr(),
            )

    @always_inline
    def dma_rope(
        self,
        k_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        w_id: Int,
        l_id: Int,
    ):
        """Writes rope cols `[ROPE_CACHE_OFFSET, ROPE_CACHE_OFFSET +
        D_ROPE)` of the latent cache row to LDS rows
        `[_K_NOPE_SLOT_ROWS, _K_NOPE_SLOT_ROWS + _K_ROPE_SLOT_ROWS)` of
        the unified `k_smem_slot`. Two details distinguish it from the
        nope DMA:

        1. The gmem source is sliced from the full-row tile at a
           column offset of `ROPE_CACHE_OFFSET // K_SUB_COLS`
           sub-blocks (=8 for FP8 K_SUB_COLS=64, =16 for BF16
           K_SUB_COLS=32), so the per-tile `.tile[K_SUB_ROWS,
           K_SUB_COLS](sub_row, sub_col + offset_sub_cols)` lands in
           the rope band of the latent row.
        2. The shared `self.k_loader.bc` is reused (one SRD for both
           segments). `SubTileLoaderLDS.load()` rebases per-call via
           `Int(src.ptr) - dram_base`, so the wide-SRD + narrow-src
           pattern is correct.
        """
        comptime _K_SUB_ROWS = Self._P._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._P._MmaOp.K_SUB_COLS
        comptime _num_kv_subblocks = (
            (Self._P.KV_BLOCK // _K_SUB_ROWS) * (Self._P.D_ROPE // _K_SUB_COLS)
        )
        comptime _warps_per_subblock = (Self._P.NUM_WARPS // _num_kv_subblocks)
        comptime _rows_per_warp = _K_SUB_ROWS // _warps_per_subblock
        comptime _num_block_cols_k = Self._P.D_ROPE // _K_SUB_COLS
        comptime _rope_sub_col_offset = Self._P.ROPE_CACHE_OFFSET // _K_SUB_COLS
        comptime assert (
            Self._P.NUM_WARPS == _num_kv_subblocks * _warps_per_subblock
        ), (
            "MlaPrefillV2Core k_rope DMA: NUM_WARPS must divide evenly"
            " into the k_rope sub-block grid"
        )
        comptime assert Self._P.ROPE_CACHE_OFFSET % _K_SUB_COLS == 0, (
            "MlaPrefillV2Core k_rope DMA: ROPE_CACHE_OFFSET must be a"
            " multiple of K_SUB_COLS so the rope band starts on a"
            " sub-block boundary"
        )

        var subblock_id, row_strip = divmod(w_id, _warps_per_subblock)
        var sub_row, sub_col = divmod(subblock_id, _num_block_cols_k)
        # rope sub-blocks land AFTER the nope sub-blocks in the unified
        # k_smem_slot — add `_K_ROPE_SUBBLOCK_OFFSET` to `subblock_id`
        # so the destination view lands in rows
        # `[_K_NOPE_SLOT_ROWS, ...)`.
        var dst_subblock_id = subblock_id + Self._P._K_ROPE_SUBBLOCK_OFFSET
        # The gmem source is the FULL latent row, so the rope sub-col
        # index must be shifted by `_rope_sub_col_offset` to address the
        # rope column band `[ROPE_CACHE_OFFSET, ROPE_CACHE_OFFSET +
        # D_ROPE)` rather than `[0, D_ROPE)`.
        var src_sub_col = sub_col + _rope_sub_col_offset

        var k_rope_src = self.k_full_gmem_tile.tile[_K_SUB_ROWS, _K_SUB_COLS](
            sub_row, src_sub_col
        ).tile[_rows_per_warp, _K_SUB_COLS](row_strip, 0)
        # KV-conditional hoist — see `dma_nope`.
        comptime if Self._HOIST:
            self.k_loader.load[hoist_scalar_offset=True](
                k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](
                    dst_subblock_id, 0
                ).tile[_rows_per_warp, _K_SUB_COLS](row_strip, 0),
                k_rope_src,
                scalar_offset=self.src_base_offset
                + (Int(k_rope_src.ptr) - Int(self.k_full_gmem_tile.ptr)),
            )
        else:
            self.k_loader.load(
                k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](
                    dst_subblock_id, 0
                ).tile[_rows_per_warp, _K_SUB_COLS](row_strip, 0),
                k_rope_src,
                scalar_offset=Int(k_rope_src.ptr)
                - self.k_loader.bc.get_base_ptr(),
            )

    @always_inline
    def dma(
        self,
        k_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        w_id: Int,
        l_id: Int,
    ):
        """Issues both nope and rope K segment DMAs into `k_smem_slot`,
        reusing the shared `KTileLoader`. Mirrors the reference's
        single-base K load: externally one call, internally two
        `buffer_load_lds` operations (DRAM has a 384-col gap between nope
        at [0, D_NOPE) and rope at [ROPE_CACHE_OFFSET, ROPE_CACHE_OFFSET
        + D_ROPE) that prevents a single contiguous DMA).

        The two underlying DMAs share `self.k_loader.k_resource` (one
        SGPR buffer-resource setup) but issue different scalar column
        offsets (0 vs ROPE_CACHE_OFFSET).
        """
        self.dma_nope(k_smem_slot, w_id, l_id)
        self.dma_rope(k_smem_slot, w_id, l_id)
