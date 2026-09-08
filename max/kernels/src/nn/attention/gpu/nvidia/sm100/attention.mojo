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
"""FA4 (Flash Attention 4) configuration for SM100 (Blackwell) kernels."""

from std.math import ceildiv, align_up, align_down, gcd
from std.sys import size_of
from std.sys import get_defined_bool
from std.bit import prev_power_of_two
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from max.gpu.primitives.grid_controls import PDLLevel
from kv_cache.types import _kv_fold_base_ok, kv_sub_tile_rows


comptime EnableForcedOrdering = get_defined_bool[
    "FA4ForcedSoftmaxOrdering", False
]()
comptime EnableEarlyAdd = get_defined_bool["FA4AddEarly", False]()

# Cross-stage P TMEM placement (2Q, non-WS only): each softmax stage's P is
# written into the OTHER stage's free S columns instead of self-aliasing its
# own S, enabling early S-release. Default OFF => byte-identical.
comptime EnableTMEMCrossP = get_defined_bool["FA4_TMEM_CROSS_P", True]()

# Reading the same define with the opposite default recovers whether the user
# passed it at all: only an explicit `=true` makes a False-defaulted read come
# back True. The cross-stage-P guards key off this, so the ON-by-default path
# can never trip a guard -- they fire only when someone explicitly asks for
# cross-stage P on a config that does not support it.
comptime ExplicitTMEMCrossP = get_defined_bool["FA4_TMEM_CROSS_P", False]()

# Programmatic Dependent Launch level for the SM100 MHA prefill kernel.  On by
# default so back-to-back attention grids in a stream overlap launch/prologue
# latency; disable with `-D MHA_PDL=false`.  When > OFF the kernel emits
# `wait_on_dependent_grids()` / `launch_dependent_grids()` and the dispatch
# attaches the PROGRAMMATIC_STREAM_SERIALIZATION launch attribute.
comptime MHA_PDL_LEVEL = PDLLevel.OVERLAP_AT_END if get_defined_bool[
    "MHA_PDL", True
]() else PDLLevel.OFF

# Bytes per CTA in shared memory that the CUDA runtime reserves for
# its own use; subtracted from `B200.shared_memory_per_multiprocessor`
# to get the usable smem budget for SM100 attention kernels.
comptime SM100_RESERVED_SMEM_BYTES = 1024


struct FA4Config[
    qkv_dtype: DType,
    *,
    rope_dtype_: Optional[DType] = None,
    scale_dtype_: Optional[DType] = None,
](TrivialRegisterPassable):
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var qk_depth: Int
    var padded_qk_depth: Int  # align_up(qk_depth, swizzle_elems)
    var ov_depth: Int
    var padded_ov_depth: Int
    # Non-rope part of the Q/K depth (= qk_depth - rope_depth). For MHA and for
    # DeepSeek-style MLA this equals `ov_depth` (V head dim == qk_nope), but for
    # GLM-style MLA where `v_head_dim != qk_nope_head_dim` they differ: the Q@K'
    # contraction and the Q_nope SMEM region are governed by `nope_depth`, while
    # the P@V output and V SMEM are governed by `ov_depth` (= v_head_dim).
    var nope_depth: Int
    var padded_nope_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
    var smem_used: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var fuse_gqa: Bool
    var swizzle_mode: TensorMapSwizzle
    var use_shared_kv: Bool
    var pair_cta: Bool
    var num_q: Int
    # Single-O TMEM mode: reuse ONE O accumulator (TMEM_O1 aliased to TMEM_O0,
    # tmem_used = 2*BN + padded_ov instead of 2*BN + 2*padded_ov), so a wide V
    # (e.g. GLM v_head_dim=256, ov_depth too big for the 2-O layout) still fits
    # the 512-col TMEM. Implies num_q==1 (the kernel body already aliases O in
    # the 1Q path). Default False ⇒ EXACTLY the pre-existing 2-O behavior (both
    # 2Q and the pre-existing prefer_1q short-seq 1Q stay byte-identical).
    var single_o: Bool
    var page_size: Int
    var is_mla: Bool
    # Split-K cluster size for the num_q==1 path: the number of CTAs grouped in
    # a cluster that partition the K/V sequence and combine via DSMEM. 1 = no
    # split-K (the cluster is then just `cta_group` for pair-CTA). Compile-time
    # because it drives the static `nvvm.cluster_dim` metadata.
    var splitk_partitions: Int
    # When True, the split-K cluster width is NOT baked into `nvvm.cluster_dim`
    # (a dynamic-cluster launch), so the per-partition load/mma/correction warps
    # must read it from `cluster_dim.x` at runtime. False -- the default, and
    # every config today, since split-K is compiled once per static P -- means
    # the width equals the comptime `splitk_partitions`, so those reads fold to a
    # constant. Retained as the single switch for a future dynamic-cluster kernel.
    var dynamic_cluster_dim: Bool
    var row_major_v_atoms: Bool
    var row_major_k_atoms: Bool
    # Warp-specialized packed-TMEM (1x4 Layout-G / 1x2 Layout-E) datapath flag
    # and its pack factor, derived in __init__ from (pair_cta, MMA_M). Stored so
    # `fa4_softmax` and other consumers read `config.use_ws` / `config.m_pack`
    # instead of re-deriving the expression. m_pack == 1 (non-WS) =>
    # byte-identical layout.
    var use_ws: Bool
    var m_pack: Int
    # When True, all `m_pack` warps of a softmax warpgroup share ONE key band:
    # the packed-TMEM accumulator quarters become DEPTH bands, so the per-WG O
    # accumulator occupies `padded_ov_depth / m_pack` physical columns instead
    # of `padded_ov_depth`. Requires `use_ws`, `num_q == 1`, and -- so that the
    # BN budget's `_o_cols` is unambiguously an O-column count rather than also
    # a KV-tile-width proxy -- MHA geometry (`not is_mla`, `nope == ov`).
    # False -- the default and every config today -- means `o_phys_cols()` is
    # literally `padded_ov_depth`, i.e. every existing config is byte-identical.
    var ws_shared_key: Bool

    # Concrete scale/rope dtypes for `Scalar[...]`/pointer reads. When the
    # optional param is unset, fall back to `qkv_dtype` so the type is always
    # well-formed; the "is it present?" signal lives in the `_size` aliases
    # below (0 when the optional is unset).
    comptime rope_dtype = Self.rope_dtype_.or_else(Self.qkv_dtype)
    comptime scale_dtype = Self.scale_dtype_.or_else(Self.qkv_dtype)

    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_dtype_size: Int = size_of[
        Self.rope_dtype_.value()
    ]() if Self.rope_dtype_ else 0
    comptime scale_dtype_size: Int = size_of[
        Self.scale_dtype_.value()
    ]() if Self.scale_dtype_ else 0

    comptime MMA_K: Int = 16 if Self.qkv_dtype.is_half_float() else 32
    comptime sm100_smem_carveout = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    @always_inline
    def BM_eff(self) -> Int:
        """Number of distinct sequence positions per full tile.
        When fuse_gqa, each tile covers BM // group seq positions x group heads.
        """
        if self.fuse_gqa:
            return self.BM // self.group
        return self.BM

    @always_inline
    def cta_group(self) -> Int:
        return 2 if self.pair_cta else 1

    @always_inline
    def cluster_size(self) -> Int:
        """CTAs per launch cluster.

        Unifies the two cluster uses: `cta_group` (pair-CTA shares one MMA
        across 2 CTAs) and `splitk_partitions` (num_q==1 split-K groups CTAs
        that independently attend over K/V partitions and combine via DSMEM).
        These are mutually exclusive today (split-K is single-CTA only), so the
        product is `cta_group` for pair-CTA and `splitk_partitions` for split-K.
        Drives the static `nvvm.cluster_dim` metadata and the launch
        `cluster_dim`.
        """
        return self.cta_group() * self.splitk_partitions

    @always_inline
    def PairBM_eff(self) -> Int:
        """Sequence positions covered by both CTAs in a pair."""
        return self.BM_eff() * self.cta_group()

    @always_inline
    def v_cols_per_cta(self) -> Int:
        """V columns stored in this CTA's SMEM."""
        if self.pair_cta:
            return self.padded_ov_depth // 2
        return self.padded_ov_depth

    @always_inline
    def v_box_cols(self) -> Int:
        """V TMA box depth (columns) per issued V load.

        WS shared sub-tile ring: V is depth-split into `num_qk_stages` 256x64
        sub-tiles, so each V TMA loads `v_cols_per_cta() // num_qk_stages`
        columns (mirrors K's `BK0`). Non-WS loads V whole (`v_cols_per_cta()`).
        MUST be used identically at EVERY V TMA type-param site (fa4_load
        signature, dispatch `create_tma_tile`, kernel launch param); a bare
        inline `... if use_ws else ...` does NOT fold to `v_cols_per_cta()` at
        parse time, so the type-param expressions would mismatch across sites.
        Routing all sites through this single method keeps them syntactically
        identical.
        """
        if self.use_ws:
            return self.v_cols_per_cta() // self.num_qk_stages
        return self.v_cols_per_cta()

    @always_inline
    def v_e_chunk_rows(self) -> Int:
        """Layout-E (`m_pack == 2`) V reduction-chunk KEY-row count.

        NOT the TMA box row count -- deliberately not named `..._box_rows`. This
        is the chunk the P@V MMA reduces over (`mma_warp`'s `pv_bk_chunk`) and
        the SMEM region one issued sub-tile lands in (`load_warp`'s
        `partition_region_elems`); `v_tma_box_rows()` splits it FURTHER when
        `page_size` is smaller, and conflating the two is what let a
        page-oblivious descriptor pair with a page-split issue loop.

        Layout-E KEY-splits V into `m_pack * num_qk_stages` per-partition
        reduction-chunk sub-tiles instead of Layout-G's DEPTH-split
        (`v_box_cols()`): each sub-tile covers `(BN // m_pack) //
        num_qk_stages` keys (this partition's OWN reduction chunk) and the
        FULL depth (`v_e_box_cols()`). Sibling of `v_box_cols()`, kept as a
        SEPARATE method (not folded into `v_box_cols()`) so Layout-G's
        `v_row_major()` / `kv_tma_fold_chunks` derivation -- which reads
        `v_box_cols()` -- stays byte-identical under the reduction-split
        geometry. Meaningful only for `m_pack == 2`; a harmless (unused)
        value otherwise.
        """
        return (self.BN // self.m_pack) // self.num_qk_stages

    @always_inline
    def v_e_box_cols(self) -> Int:
        """Layout-E (`m_pack == 2`) V TMA box depth (columns) per issued
        sub-tile: the FULL `padded_ov_depth`, since Layout-E splits V by KEY
        (reduction), not by depth. Counterpart of `v_e_chunk_rows()`.

        This one DOES keep the `box` name: paging partitions the KEY axis only,
        so unlike the row count this is the descriptor's column count outright,
        with nothing further to split.
        """
        return self.padded_ov_depth

    @always_inline
    def v_tma_box_rows(self, page_size: Int) -> Int:
        """V TMA box KEY-row count for this config's layout.

        The single selector every V TMA type-param site routes through
        (dispatch `create_tma_tile`, kernel `VTMAOpType`, the `fa4_load`
        signature), so the box shape stays syntactically identical across
        sites -- the same single-source-of-truth rule `v_box_cols()`
        documents.

        One composition, two row sources: Layout-E (`m_pack == 2`) starts from
        its KEY-split reduction chunk `v_e_chunk_rows()`, Layout-G / non-WS from
        the whole tile `BN`; BOTH are then split by paging. The paging split
        does not substitute for the reduction split -- `v_e_chunk_rows()` is a
        MATH partition (which keys one P@V MMA reduces) while `page_size` is a
        PHYSICAL address discontinuity, so when `page_size < v_e_chunk_rows()` a
        reduction chunk straddles pages and must be cut again.
        `_tma_copy_kv_impl` re-derives that same cut as its per-issue row count
        (`kv_sub_tile_rows(tile_rows, page_size)`) and issues `pages_per_iter`
        TMAs of the descriptor's box, so a box built without the paging term
        would over-deliver by exactly that factor and desync the KV ring's
        `expect_tx` accounting. `kv_sub_tile_rows` is the identity whenever
        `page_size >= rows`, so Layout-G / non-WS stay byte-identical to the
        historical inline expression.

        **Shared-key is a THIRD row source, and it is Layout-E's shape without
        the partition split.** V is cut along its own *reduction* axis (keys),
        never along depth: one slot is then exactly one P@V MMA's B operand, so
        V liveness is 1 in-flight slot instead of the `pv_mma_n() /
        kv_sub_depth()` depth slices Layout-G must hold simultaneously. That is
        what makes the deep bf16 cells fit -- see `ring_slots_needed()`. The row
        count is `pv_key_chunk()`, which already computes exactly this quantity
        (`(BN * BK0) / pv_mma_n()`), so the ring's K/V byte-equality is
        definitional here rather than an algebraic coincidence to be checked:
        `pv_key_chunk() * pv_mma_n() == BN * BK0` by construction.
        """
        if self.ws_shared_key:
            return kv_sub_tile_rows(self.pv_key_chunk(), page_size)
        var rows = self.v_e_chunk_rows() if self.m_pack == 2 else self.BN
        return kv_sub_tile_rows(rows, page_size)

    @always_inline
    def v_tma_box_cols(self) -> Int:
        """V TMA box depth (columns) for this config's layout.

        Sibling of `v_tma_box_rows()`: Layout-E (`m_pack == 2`) uses the
        full-depth `v_e_box_cols()`, Layout-G / non-WS the depth-split
        `v_box_cols()`.

        Shared-key uses `pv_mma_n()`, NOT `v_e_box_cols()`'s
        `padded_ov_depth`. The two coincide at depth 256 and differ at 512,
        where `pv_mma_n()` saturates at 256 and the warpgroup walks
        `num_o_tiles()` output depth tiles: the box must be ONE tile's depth,
        because a box spanning both tiles would make a slot hold V for an
        accumulator the current MMA is not writing -- reinstating exactly the
        multi-slot liveness this layout exists to remove. Spelling it
        `padded_ov_depth` would be right at d256 and silently wrong at d512.
        """
        if self.ws_shared_key:
            return self.pv_mma_n()
        if self.m_pack == 2:
            return self.v_e_box_cols()
        return self.v_box_cols()

    @always_inline
    def v_tma_tile_rows(self) -> Int:
        """V TMA *tile* row stride at the issue site (`BN // num_v_sub_tiles`),
        NOT the box row count -- feeding the fold's `box_rows == tile_rows`
        test the box rows makes it vacuously true; feeding it `BN` pinned the
        shared-key fold to 1.

        Shared-key: `pv_key_chunk()`. Layout-E: `v_e_chunk_rows()`. Layout-G /
        non-WS: `BN`.
        """
        if self.ws_shared_key:
            return self.pv_key_chunk()
        if self.m_pack == 2:
            return self.v_e_chunk_rows()
        return self.BN

    @always_inline
    def nope_cols_per_cta(self) -> Int:
        """K_nope columns stored in this CTA's SMEM (per-CTA padded nope width).

        Sibling of `v_cols_per_cta()` on `padded_nope_depth`. Equals
        `v_cols_per_cta()` for MHA / DeepSeek (nope == ov).
        """
        if self.pair_cta:
            return self.padded_nope_depth // 2
        return self.padded_nope_depth

    @always_inline
    def shared_kv_cols(self) -> Int:
        """Un-halved width of one shared K_nope/V SMEM stage.

        K_nope (padded_nope_depth) and V (padded_ov_depth) share one buffer, so
        a stage fits the wider of the two. This is the *full* (non-pair-halved)
        column count; pair-CTA halving is applied at the call site where needed.
        Equals `padded_ov_depth` for MHA / DeepSeek (nope == ov).
        """
        return max(self.padded_nope_depth, self.padded_ov_depth)

    @always_inline
    def pv_partitions(self) -> Int:
        """Independent key partitions the P@V accumulator of one warpgroup holds.

        Default (`ws_shared_key == False`): each of the `m_pack` packed-TMEM
        warps is its own key partition carrying a full-depth O partial, so this
        is `m_pack` (and 1 on the non-WS path, where `m_pack == 1`). Under
        shared-key mode the whole warpgroup walks one key band, so there is a
        single partition and the quarters become depth bands instead.
        """
        return 1 if self.ws_shared_key else self.m_pack

    @staticmethod
    @always_inline
    def _o_phys_cols(
        padded_ov_depth: Int, m_pack: Int, ws_shared_key: Bool
    ) -> Int:
        """Physical TMEM columns one O accumulator occupies.

        A `@staticmethod` on purpose: `__init__` needs this quantity while
        `self` is still partially initialized, and a *method* call there would
        borrow all of `self` and be rejected. Because this borrows nothing,
        `__init__` and `o_phys_cols()` can share ONE spelling instead of
        maintaining two that a test has to bind together.

        Spelled with an explicit mode-off branch rather than as
        `pv_partitions() * (padded_ov_depth // m_pack)` so the identity is
        *syntactic*: the mode-off arm is literally `padded_ov_depth` and folds
        without the compiler having to prove `m_pack * (x // m_pack) == x`.
        That algebraic form is also silently wrong for any `padded_ov_depth`
        that is not a multiple of `m_pack`.
        """
        if ws_shared_key:
            return padded_ov_depth // m_pack
        return padded_ov_depth

    @always_inline
    def o_phys_cols(self) -> Int:
        """Physical TMEM columns one O accumulator occupies.

        Thin forward to `_o_phys_cols`, which `__init__` also calls -- the two
        cannot drift.
        """
        return Self._o_phys_cols(
            self.padded_ov_depth, self.m_pack, self.ws_shared_key
        )

    @always_inline
    def ws_epilogue_stage_f32(self) -> Int:
        """Per-warpgroup Level-1 raw-O staging, in f32 slots.

        `m_pack * BM * ov_depth` is **128 KiB per warpgroup** at Layout-G /
        depth 256, and both warpgroups carve one. That, plus the L2 staging and
        the output, is ~305 KiB against a 232 KiB carveout -- i.e. today's
        epilogue cannot run at these depths at all, and the budget assert in
        `fa4_softmax` says so.

        Shared-key mode removes it entirely: with one key band and one shared
        accumulator there is no `m_pack`-way fold, so there is nothing to
        stage. `fa4_ws_level0_band` reads each warp's band straight out of TMEM
        into registers. That is what makes the epilogue fit at depth 256/512.

        On the config rather than at the carve because the offsets are computed
        INDEPENDENTLY in two places (the T>=2 epilogue and the empty-partition
        split-K path), which must keep their DSMEM offsets aligned across
        partitions. Mode-independent duplicates agree by inspection; mode-dependent
        ones would not.
        """
        if self.ws_shared_key:
            return 0
        return self.m_pack * self.BM * self.ov_depth

    @always_inline
    def ws_epilogue_ml_f32(self) -> Int:
        """Per-warpgroup Level-1 `(m, l)` staging, in f32 slots.

        Zero under shared-key for the same reason as the O staging: the row max
        is already agreed by the per-iteration exchange and the row sum goes
        through the dedicated `ws_exchange` region, so no `(m, l)` ever transits
        this one.
        """
        if self.ws_shared_key:
            return 0
        return self.m_pack * self.BM * 2

    @always_inline
    def ws_epilogue_f32_slots(self) -> Int:
        """Total f32 slots the WS epilogue carves from the dead Q+KV span.

        `2 * (stage + ml)` for the two warpgroups' Level 1, plus the single
        cross-WG L2 staging and its `(m_1, l_1)`. The output tile is NOT
        included -- it is `output_type`, not f32, and the caller adds it.

        The L2 terms are mode-independent: `ov_depth * BM` covers all
        `num_o_tiles()` per-tile windows exactly, because the windows partition
        the depth. So the tiled shared-key epilogue needs no more L2 staging
        than the single-call one, only a different subdivision of it.

        "Exactly" is conditional, and the condition is enforced elsewhere. The
        windows sum to `num_o_tiles() * pv_mma_n() * BM`, which is
        `padded_ov_depth * BM` by `num_o_tiles()`'s defining invariant -- the
        PADDED depth, while this term is the unpadded one. `supported()`
        conjunct (5) rejects shared-key whenever the two differ, so do not read
        this docstring as proof that any `ov_depth` works.
        """
        return (
            2 * self.ws_epilogue_stage_f32()
            + 2 * self.ws_epilogue_ml_f32()
            + self.ov_depth * self.BM
            + self.BM * 2
        )

    @always_inline
    def correction_o_cols(self) -> Int:
        """TMEM columns `correction_warp._rescale_o` must walk per accumulator.

        NOT a synonym for `o_phys_cols()`, and deliberately not spelled as one:
        the mode-off arm is the **unpadded** `ov_depth`, which is what
        `_rescale_o` has always used and is narrower than `padded_ov_depth`
        wherever the swizzle pads (the `test_mla_prefill_vhead_repro_ps*`
        shapes). Substituting the padded width would rescale columns the
        producer never wrote.

        Mode-on the accumulator is packed `m_pack` ways into depth bands, so
        the region really is `o_phys_cols()` columns wide. Walking the logical
        `ov_depth` instead would run 4x past it at d256 -- over `TMEM_O1` and
        the `S` windows, then past the CTA's TMEM allocation entirely. That
        fault is silent: wrong answers, no trap.

        `supported()` gates on this so the `comptime assert load_iters > 1`
        inside `_rescale_o` can never be violated by a geometry the dispatcher
        accepted -- one spelling, checked once.
        """
        if self.ws_shared_key:
            return self.o_phys_cols()
        return self.ov_depth  # mode-off arm stays LITERAL (unpadded)

    @always_inline
    def pv_mma_n(self) -> Int:
        """`MMA_N` of the P@V MMA.

        The mode-off arms reproduce `mma_warp.mojo`'s three-way ladder
        *verbatim*, deliberately not unified: Layout-E's
        `m_pack * padded_ov_depth` and Layout-G's `m_pack * (256 // m_pack)`
        are different expressions that merely coincide at some shapes, and the
        non-WS arm is a third. A "simplification" here is a silent geometry
        change on a shipping path.

        This is also the exit for the two *inert* duplicate declarations of the
        same MMA (`softmax_warp.mojo`, `kernel.mojo`), which spell `MMA_N` as
        `padded_ov_depth` and disagree with `mma_warp` on both WS layouts.
        """
        if self.ws_shared_key:
            # One MMA per output depth tile; tcgen05's N is capped at 256.
            return min(self.padded_ov_depth, 256)
        if self.use_ws and self.m_pack == 2:  # Layout-E, reduction split
            return self.m_pack * self.padded_ov_depth
        if self.use_ws:  # Layout-G, depth scatter
            return self.m_pack * (256 // self.m_pack)
        return self.padded_ov_depth  # non-WS

    @always_inline
    def num_o_tiles(self) -> Int:
        """Output depth tiles the P@V walks -- `mma_warp`'s C-stride count.

        `TMEM_O0`/`TMEM_O1` are the two Q-stage pipelines, not two tiles of one
        warpgroup. Mode-off this counts the depth tiles a Layout-G P@V scatters
        O across (2 at shipping d128); mode-on it is the number of
        accumulators, because `pv_mma_n()` saturates at 256 and a depth beyond
        that needs a second one. Both readings are the same division, which is
        why one accessor serves them.

        Spelled against `pv_mma_n()` rather than a literal 256. The two are equal
        at every value this derives today (64/128/256/512), but they are
        *different quantities* -- the tile count vs. the tile width -- and the
        shared-key epilogue walks `num_o_tiles()` tiles of
        `pv_mma_n() // m_pack` columns each, so a disagreement scatters O to the
        wrong depth columns silently. The binding invariant is now the
        DEFINITION, not a separate claim a test has to bind:
        `num_o_tiles() * (pv_mma_n() // m_pack) == o_phys_cols()`. The
        `pv_mma_n() // m_pack` factor is the per-warp band width
        (`softmax_warp`'s `band_cols`) and is also exactly `depth_tile` in every
        mode (Layout-G `256/4`, Layout-E `256/2`, non-WS `padded_ov_depth`),
        so one expression serves the mode-off depth-tile count and the mode-on
        accumulator count.

        `ceildiv`, not floor: mode-on this must stay equal to
        `ceildiv(padded_ov_depth, pv_mma_n())`, which floor division would break
        at any depth that is not a multiple of the band (a padded 384 gives 2;
        floor would give 1 and drop a tile).

        NOTE: `mma_warp`'s local `num_d_tiles` is `num_qk_stages` (2) there,
        because it counts REDUCTION chunks, not depth tiles -- a different
        quantity wearing the same name (see U3). Do not "fix" one row to match
        the other.
        """
        return ceildiv(self.o_phys_cols(), self.pv_mma_n() // self.m_pack)

    @always_inline
    def pv_key_chunk(self) -> Int:
        """Keys contracted by ONE P@V MMA.

        Mode-on this is forced by the ring geometry, not chosen: the B operand
        of one MMA must exactly fill one KV ring slot, which is the byte
        identity `mma_warp.mojo:480-486` already asserts for Layout-E. With
        `sub_depth == BK0` (MHA, which the shared-key guard requires) that
        reduces to `BN * BK0 / pv_mma_n`.

        Mode-off keeps the literal ladder. The unified formula is NOT valid
        there: on the non-WS arm it needs `BK0 == padded_ov_depth`, which
        fails for every config with `num_qk_stages > 1`.
        """
        if self.ws_shared_key:
            return (self.BN * self.BK0) // self.pv_mma_n()
        if self.use_ws and self.m_pack == 2:  # Layout-E
            return self.v_e_chunk_rows()
        if self.use_ws:  # Layout-G
            return self.BN // self.m_pack
        return self.BN  # non-WS

    @always_inline
    def pv_reduction_chunks(self) -> Int:
        """Chunks the shared-key P@V **reduction** axis is cut into.

            pv_reduction_chunks() * pv_key_chunk() == BN      # 4 * 64 == 256

        P@V has two INDEPENDENT splits, and the names must keep them apart:

            depth      the MMA's N axis   `num_o_tiles()`   1 @ d256, 2 @ d512
            reduction  the MMA's K axis   this accessor     4 @ d256, 4 @ d512

        Their product is `v_ring_positions_per_tile()`, the ring positions one
        V tile occupies. Named for the AXIS rather than as a "per tile" count:
        "tile" is ambiguous here, since the chunk set is per KV tile and repeats
        identically for each of the `num_o_tiles()` depth tiles.

        The OUTER axis of the 2-D shared-key V walk, and the axis a short final
        tile truncates. Both warps derive their trip count from this one
        accessor: the producer emits chunks `[0, live)` and the consumer
        contracts the same `[0, live)`, so a divergence here is a hang rather
        than a wrong answer.

        Chunk-outer / depth-tile-inner is deliberate. A dead chunk then removes
        a CONTIGUOUS run of `num_o_tiles()` positions from the END of the walk,
        which keeps the live set a prefix; the transposed order would interleave
        the skips and force both sides to reconstruct a non-contiguous pattern.
        It also lets the producer hoist its per-chunk `populate` out of the
        depth-tile loop, since all `num_o_tiles()` tiles of a chunk read the
        SAME key rows.

        `supported()` requires `BN % pv_key_chunk() == 0`, so this is exact --
        a ragged last chunk would desynchronize the two sides' `c * KC`
        arithmetic from `BN` silently.
        """
        return self.BN // self.pv_key_chunk()

    @always_inline
    def v_ring_positions_per_tile(self) -> Int:
        """KV ring positions ONE V tile occupies under the shared-key walk.

        The walk is 2-D -- `num_o_tiles()` output depth tiles x
        `pv_reduction_chunks()` key chunks -- and this is their product. It is
        the consumer-side twin of the producer's `_emit_v` trip count
        (`load_warp.mojo`, `comptime for stage in range(config.num_qk_stages)`),
        so the two agreeing is what keeps the ring in step.

        **They agree by ALGEBRA, not by coincidence.** Substituting
        `pv_key_chunk() == BN*BK0 / pv_mma_n()` and
        `num_o_tiles() == padded_ov_depth / pv_mma_n()`:

            num_o_tiles * (BN / pv_key_chunk)
              = (padded_ov / pv_mma_n) * (BN * pv_mma_n) / (BN * BK0)
              = padded_ov / BK0
              = num_qk_stages

        so the identity holds for every shared-key geometry, not just the two we
        ship. That is why `supported()` asserts THIS rather than the older
        `num_o_tiles() == num_qk_stages`, which is the same statement only when
        there is exactly one key chunk and is simply false at BN=256 (d256 gives
        1 != 4, d512 gives 2 != 8).

        Meaningless off the shared-key walk -- Layout-G's V positions are depth
        tiles and Layout-E's are reduction chunks, neither of which is a product
        of these two factors. Guard call sites on `ws_shared_key`.
        """
        return self.num_o_tiles() * self.pv_reduction_chunks()

    @always_inline
    def v_pages_per_chunk(self, page_size: Int) -> Int:
        """Page slots one shared-key V key chunk's TMA box spans.

        1 whenever `page_size >= pv_key_chunk()` (64), because
        `kv_sub_tile_rows` is the identity there -- so at the production paged
        shapes {64, 128} a key chunk is exactly ONE TMA issue and `valid_pages`
        is binary: either the chunk is fully issued or it is the fully-empty TMA
        the walk skips. It only exceeds 1 at `page_size < 64`.
        """
        return self.pv_key_chunk() // kv_sub_tile_rows(
            self.pv_key_chunk(), page_size
        )

    @always_inline
    def v_oob_fill_needed(self, page_size: Int) -> Bool:
        """Whether a partial V sub-tile must OOB-zero-fill its dead page slots.

        Zero-fill exists to make an UNCONDITIONALLY-ISSUED P@V read finite data.
        It is needed for either of two independent reasons:

        1. **The position cannot be dropped** -- `pv_partitions() > 1`. When one
           ring position serves several independent key partitions, one
           partition's chunk can be dead while another's is live, so every pack
           takes the same cut and the position must be filled rather than
           skipped. This is Layout-E's case, argued at `load_warp.mojo`'s
           `_produce_v_e`, and Layout-G's (its V positions are depth slices
           spanning all BN keys, so only their tail pages are dead).
        2. **A LIVE position has dead page slots inside it** --
           `v_pages_per_chunk() > 1`. Only reachable at `page_size < 64`.

        Under shared-key at the production page sizes neither holds: there is
        one partition, so a dead key chunk is dead for every consumer of that
        position and producer and consumer skip it in lockstep; and a live chunk
        is one whole TMA issue with nothing dead inside it. Nothing is left to
        fill -- which is the point, since filling it would mean writing a slot
        of zeros so the MMA warp can multiply by them.

        Spelled on `pv_partitions()` rather than `not ws_shared_key` because the
        partition count is the actual reason, and stays right if another
        multi-partition layout appears. Reproduces today's `partial and use_ws`
        selector exactly off the shared-key path: `m_pack == 1` iff not
        `use_ws`, so `pv_partitions() > 1` is precisely `use_ws` there.
        """
        return self.pv_partitions() > 1 or (
            self.ws_shared_key and self.v_pages_per_chunk(page_size) > 1
        )

    @always_inline
    def kv_sub_depth(self) -> Int:
        """Depth covered by one KV ring sub-tile.

        The `sub_depth` local `__init__` uses to size `bytes_per_subtile`.
        """
        return self.shared_kv_cols() // self.num_qk_stages

    @always_inline
    def v_residency(self) -> Int:
        """`R_V`: the most V ring positions the consumer holds at once.

        The only non-zero term of `ring_slots_needed()`'s span decomposition on
        the co-rotated schedule -- see there for why `H_K` and `W_K` vanish.
        Split out so the two release disciplines are visibly different
        quantities rather than one expression that happens to cover both.

        **Key-split (mode off): the whole V tile.** `_pv_ws`'s legacy arm waits
        every sub-slot up front (`_v_wait_rest`), MMAs them all, then releases the
        group (`_v_release_first` + `_v_release_rest`). The residency is the
        sub-tile count a V tile occupies, which equals `num_qk_stages` wherever
        `mma_warp`'s `v_subslots == num_qk_stages` assert holds. Keep the
        `padded_ov_depth // kv_sub_depth()` spelling -- the quantity the
        derivation bounds, reporting truth rather than a coincidence on a
        layout that breaks that assert.

        **Shared-key: ONE.** The key-chunk walk releases each position the
        instant its MMAs are issued (`mma_warp`'s `_commit` at the foot of the
        `(chunk, depth tile)` body). `_commit` lowers to `tcgen05.commit`, so
        the arrival is ordered BEHIND every MMA the warp issued and the release
        cannot outrun the operand read. That is the whole reason the key-chunk
        layout exists, and what makes bf16 d512 expressible at all: a whole K
        tile there is `8 * 32768 B`, more than the 232448 B carveout, so no
        whole-tile residency can fit.

        **A claim about the CONSUMER, true only while every KV-slot reader is
        an async MMA.** Give a slot a non-MMA reader -- LDSM, `ld.shared`,
        `cp.async` -- and release-after-issue stops being ordered by
        `tcgen05.commit`: the producer may refill a slot the reader has not
        drained. Silent corruption, not a hang. P is staged through `p_smem`,
        carved and charged separately, precisely so this stays true.
        """
        if self.ws_shared_key:
            return 1
        return self.padded_ov_depth // self.kv_sub_depth()

    @always_inline
    def ring_slots_needed(self) -> Int:
        """KV ring slots the P@V schedule must have to make progress.

        Deadlock-freedom bound consumed by `supported()`, not a performance
        target. Under-tightening does not fail to compile and gives no wrong
        answer -- it HANGS, consumer blocked in `consumer_wait()` (`mma_warp`)
        and producer in `producer_acquire()` (`load_warp`).

        A SPAN over ring positions, not a count of live slots.
        `producer_acquire` waits on `consumer_mbar()` keyed by the producer's
        own `state.index()`, and both warps walk one cursor one step at a
        time, so the producer may fill position `q` only after the consumer
        released `q - N`. A released-but-not-yet-recycled position still costs
        a slot:

            N_min = max over consumer waits at q of
                      (q - oldest_unreleased(q) + 1)

        Under WS every K and V depth sub-tile is its OWN ring position, so

            N_min = R_V + H_K + W_K

        with `R_V` the most V positions held at once, `H_K` one if the
        preceding K tile's last sub-slot is still unreleased when the V walk
        blocks on its d0, `W_K` a whole K group if a V group stays held across
        the K traversal. On the co-rotated 1Q schedule both K terms vanish:
        `W_K = 0` (rotation consumes in emission order, so no V group spans a
        K traversal) and `H_K = 0` (three sites in `mma_warp`'s `_body_1q`
        retire the preceding K tile's LAST sub-slot before the next V d0 wait;
        remove any one and `H_K` becomes 1, returning the floor to `S + 1`). K's
        own `num_qk_stages` extent is a DISTANCE TRAVERSED, not residency.

        Spelled `v_residency()` not `num_qk_stages` so the two release
        disciplines stay two quantities (key-split holds a whole V tile,
        shared-key one position).

        Must be EXACTLY the deadlock bound, with no slack: bf16 d512 carries 4
        slots and cannot get a fifth, so a `+1` -- inert on every other cell --
        would silently reject it. Slack is a pipelining concern
        (`run_ahead = num_kv_stages - N_min`), not correctness.

        This accessor and the schedule are ONE unit: this bound against an
        unrotated producer/consumer hangs any config with `num_kv_stages` in
        [S, 2*S), so the rotation cannot be disabled.
        """
        if not self.use_ws:
            # Non-WS: `num_qk_stages` is forced to 1 (`__init__`'s shared arm)
            # and one slot holds a whole tile, so K and V contribute one
            # position each. Matches `supported()`'s base `>= 2` conjunct.
            return 2
        # Rotated schedule: `R_V` alone (see the decomposition above).
        return self.v_residency()

    @always_inline
    def ws_arena_slots_needed(self) -> Int:
        """The OTHER floor on `num_kv_stages`: the WS combine epilogue's arena.

        `num_kv_stages` is not only a ring depth. The WS two-level combine
        carves its f32 staging out of the dead Q+KV span and requires
        (`softmax_warp.mojo`)

            ws_epilogue_f32_slots()*4 + BM*ov_depth*2  <=  q_bytes + kv_bytes

        and on the WS path `kv_bytes == num_kv_stages * subtile_bytes`. So that
        is a lower bound on the stage count, independent of
        `ring_slots_needed()` -- and the LARGER of the two on every cell, so
        the ring is nowhere the binding constraint.

        Which one binds flips with the mode: shared-key drives
        `ws_epilogue_stage_f32()`/`ws_epilogue_ml_f32()` to 0, and that collapse
        is what makes depth 256/512 fit at all.

        **Why this is a `supported()` conjunct and not left as the assert.**
        Every other admission constraint reaches dispatch as a graceful
        `comptime if` fall-through (`dispatch.mojo`), but the epilogue assert is
        a `comptime assert` deep inside `fa4_softmax`, so violating it is a
        **build error, not a reroute** -- and `bm64` d128 sits EXACTLY on its
        floor, so any new fixed SMEM region trips it. That assert stays as a
        redundant backstop. The layout here is duplicated from `smem.mojo`
        (importing it would be circular); `test_fa4_config_scaffold` binds the
        two spellings with an equality assert.

        The `2` is `size_of[output_type]()`, which `supported()` cannot see (a
        kernel param). WS requires a 2-byte output, so 2 is exact on every
        config that can reach the epilogue; a wider output would make the real
        arena larger than this estimate, caught by the backstop assert.
        """
        if not self.use_ws:
            # Non-WS prunes the whole epilogue block, so it imposes no floor.
            return 0
        var subtile_bytes = (
            self.BN
            * (self.shared_kv_cols() // self.num_qk_stages)
            * Self.qkv_dtype_size
        )
        if subtile_bytes <= 0:
            # Degenerate cell (BN solves to 0 or negative at depths this layout
            # cannot host). Impose nothing and let the conjuncts that actually
            # describe the failure do the rejecting, rather than masking them
            # behind a bound computed from nonsense.
            return 0
        var arena_bytes = (
            self.ws_epilogue_f32_slots() * size_of[DType.float32]()
            + self.BM * self.ov_depth * 2
        )
        var q_bytes = (
            self.BM * self.padded_nope_depth * Self.qkv_dtype_size
            + self.BM * self.rope_depth() * Self.rope_dtype_size
        )
        if arena_bytes <= q_bytes:
            return 0
        return ceildiv(arena_bytes - q_bytes, subtile_bytes)

    @staticmethod
    @always_inline
    def _ws_exchange_bytes(ws_shared_key: Bool) -> Int:
        """Bytes of the shared-key cross-warp row-max exchange region.

        `2 (parity) * 2 (warpgroup) * WARPGROUP_SIZE` Float32 slots = 2 KiB.

        Fixed by the number of *softmax threads*, not by `BM` or `m_pack`:
        every one of a warpgroup's 128 threads deposits one partial then re-reads
        all `m_pack` of them. `m_pack * BM == WARPGROUP_SIZE` makes the slot map
        work (thread `t` owns row `t % BM` of warp `t / BM`), and `supported()`
        REJECTS a shared-key config that breaks it rather than leaving it a
        coincidence -- the correction region carries the same warning for the
        same reason.

        Double-buffered on iteration parity so ONE `named_barrier` per exchange
        suffices (see `fa4_ws_exchange4` for the race argument). A dedicated
        region, not a slice of `correction_smem`, precisely because the
        read-vs-correction-write overlap there is what forces the depth512
        ancestor's second barrier.

        A `@staticmethod` so `__init__` (partially-initialized `self`) and
        `SM100AttentionSMem` share ONE spelling -- the two *must* agree exactly,
        and `mla_prefill_blockscale.mojo` asserts `smem_size() == smem_used`.
        """
        if ws_shared_key:
            return 2 * 2 * WARPGROUP_SIZE * size_of[DType.float32]()
        return 0

    @always_inline
    def ws_exchange_bytes(self) -> Int:
        """Bytes of the shared-key cross-warp exchange region (0 when off)."""
        return Self._ws_exchange_bytes(self.ws_shared_key)

    @staticmethod
    @always_inline
    def _p_smem_bytes(BM: Int, BN: Int, ws_shared_key: Bool) -> Int:
        """Bytes of the shared-key P staging region.

        Under shared-key mode the P@V MMA becomes an SS MMA whose A operand is
        the full `BM x BN` P tile spanning all `m_pack` warps, so P can no
        longer live in each warp's own packed-TMEM S window. One tile per
        softmax warpgroup: both warpgroups are concurrently live (only
        `single_o` / `total_iters <= 1` parks WG1, and reserving the second
        buffer there is merely unused, never wrong).

        Single-buffered per warpgroup on purpose: RAW is covered by
        `consumer_s`, WAR by the existing `pipeline_s` producer barrier via
        UMMA issue order (see the `fa4_ws_exchange4` docstring).
        """
        if ws_shared_key:
            return 2 * BM * BN * Self.qkv_dtype_size
        return 0

    @always_inline
    def p_smem_bytes(self) -> Int:
        """Bytes of the shared-key P staging region (0 when off)."""
        return Self._p_smem_bytes(self.BM, self.BN, self.ws_shared_key)

    @always_inline
    def k_rows_per_cta(self) -> Int:
        """K rows stored in this CTA's SMEM."""
        if self.pair_cta:
            return self.BN // 2
        return self.BN

    @always_inline
    def v_row_major(self) -> Bool:
        """Effective row-major (page-dense, chunk-inner) V layout selector.

        Drives BOTH the V TMA producer fold (`tma_copy_v[row_major=...]`) and
        the P@V MMA consumer descriptor (`smem_descriptor[page_dense=...]` /
        `SM100TensorAccumulator[b_page_dense=...]`); a single accessor keeps the
        producer's page-dense SMEM and the consumer's descriptor in agreement.

        Returns True only when the page-dense layout is actually applicable —
        i.e. when the V-side `kv_tma_fold_chunks[row_major=True]` would fold
        (`>= 2`). The `base_ok` geometry comes from the SHARED `_kv_fold_base_ok`
        gate (the single source of truth the predicate also uses), so the two
        cannot drift; a `comptime assert` at the dispatch site still cross-checks
        this accessor against the real fold result (it spans the
        `box_rows == page_size` bridge the shared gate does not).

        Restricted to genuine multi-page paging (`0 < page_size < BN`):
          - This is the only regime where the fold helps: with `page_size >= BN`
            or `page_size == 0` the tile is a single page (`pages_per_iter == 1`)
            that the chunk-outer rank-4 fold already loads in one TMA.
          - It is also the only regime where the rank-5 atom-row coordinate
            (`gmem_row // _SWIZZLE_ATOM_ROWS`) is safe: a block-indirected
            `PagedKVCache` gives `gmem_row = block_idx * stride` with `stride` a
            multiple of `page_size`, so `page_size % 8 == 0` (checked below)
            makes every page row 8-aligned. Continuous / ragged operands
            (`page_size >= BN`) place tiles at arbitrary token offsets that are
            NOT 8-aligned, which would corrupt the atom-row coordinate.

        Gated to single-CTA: under `pair_cta` the descriptor (`BMN =
        v_cols_per_cta() = ov/2`) and the accumulator advance (`b_BMN = MMA_N =
        ov`) disagree on `mn_dim` for the native layout — a fast-follow change.
        SWIZZLE_128B only (the native layout is defined there).
        """
        if not self.row_major_v_atoms:
            return False
        if self.swizzle_mode != TensorMapSwizzle.SWIZZLE_128B:
            return False
        if self.pair_cta:
            return False
        # Multi-page paging only (see docstring): single-page / continuous /
        # ragged stay chunk-outer.
        if not (self.page_size > 0 and self.page_size < self.BN):
            return False
        var gran = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        # base_ok: shared with kv_tma_fold_chunks (single source of truth) —
        # BK % gran == 0, >= 2 chunks, head_size (= ov_depth) divisible by BK.
        # Use the PER-TMA V depth `v_box_cols()` (the actual `BK` handed to the
        # V-side kv_tma_fold_chunks in dispatch), mirroring how `k_row_major()`
        # uses `BK0`. It equals `v_cols_per_cta()` for non-WS (V loaded whole ->
        # byte-identical), but for the WS shared sub-tile ring it is
        # `v_cols_per_cta() // num_qk_stages == gran` (one gran-chunk per V
        # sub-tile), so base_ok is False -> row-major does not apply and the
        # producer/consumer both take the chunk-outer layout, exactly like K's
        # `BK0 == gran` non-shared-KV case. Using `v_cols_per_cta()` here would
        # over-report foldability and drift from the real per-sub-tile fold.
        if not _kv_fold_base_ok(self.v_box_cols(), gran, self.ov_depth):
            return False
        # geometry_ok (row_major): box_rows == page_size here (page_size < BN),
        # so the TMA sub-tile must split into _SWIZZLE_ATOM_ROWS (= 8) atom-rows.
        # This also guarantees the gmem page rows are 8-aligned (see docstring).
        return self.page_size % 8 == 0

    @always_inline
    def k_row_major(self) -> Bool:
        """Effective row-major (page-dense, chunk-inner) K layout selector.

        K-side analog of `v_row_major()`: drives BOTH the K TMA producer fold
        (`tma_copy_k[row_major=...]`) and the Q@K' MMA consumer descriptor
        (`smem_descriptor[is_k_major=True, page_dense=...]` /
        `SM100TensorAccumulator[b_page_dense=...]` for the k-major B operand).
        A single accessor keeps the producer's page-dense SMEM and the
        consumer's descriptor in agreement (disagreement is silent wrong
        output, not a crash).

        Gated identically to V: `SWIZZLE_128B` only, single-CTA only
        (`not pair_cta`), and genuine multi-page paging
        (`0 < page_size < k_rows_per_cta()` with `page_size % 8 == 0`, so a
        block-indirected `PagedKVCache` gives 8-aligned page rows for the
        rank-5 atom-row coordinate — see `v_row_major()` for the full
        rationale). The `base_ok` geometry comes from the SHARED
        `_kv_fold_base_ok` gate (the K-side `kv_tma_fold_chunks` uses the same
        gate: `BK0 % gran == 0`, `num_chunks >= 2`, `qk_depth % BK0 == 0`), so
        the two cannot drift; a `comptime assert` at the dispatch site still
        cross-checks this against the real fold result.

        K and V share the same constructor-computed default
        (`row_major_{v,k}_atoms = not is_mla and 0 < page_size < BN`); the two
        accessors then apply their own feasibility gates independently.
        """
        if not self.row_major_k_atoms:
            return False
        if self.swizzle_mode != TensorMapSwizzle.SWIZZLE_128B:
            return False
        if self.pair_cta:
            return False
        # Multi-page paging only (see v_row_major docstring): single-page /
        # continuous / ragged stay chunk-outer. `k_rows_per_cta()` is the K
        # tile's seq_k extent (== BN single-CTA, which is enforced above).
        if not (self.page_size > 0 and self.page_size < self.k_rows_per_cta()):
            return False
        var gran = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        # base_ok: shared with the K-side kv_tma_fold_chunks (single source of
        # truth) — BK0 % gran == 0, >= 2 chunks, qk_depth divisible by BK0.
        #
        # The `>= 2 chunks` term naturally restricts the fold to the regime where
        # it helps: in the non-WS shared-KV mode `num_qk_stages == 1` so
        # `BK0 == padded_qk_depth`
        # (multiple gran-chunks per K tile to fold). In non-shared-KV mode `BK0 == gran`
        # (one gran-chunk per stage, separate buffers), so there is one chunk and
        # this returns False — correct, since K already loads one TMA per page
        # per stage there (nothing to fold).
        if not _kv_fold_base_ok(self.BK0, gran, self.qk_depth):
            return False
        # geometry_ok (row_major): box_rows == page_size here (< k_rows_per_cta),
        # so the TMA sub-tile splits into _SWIZZLE_ATOM_ROWS (= 8) atom-rows and
        # the gmem page rows are 8-aligned.
        return self.page_size % 8 == 0

    @always_inline
    def q_nope_bytes(self) -> Int:
        """Q nope region bytes: BM * padded_nope_depth * dtype_size.

        The Q_nope tile feeds the Q@K_nope' contraction, so its width is the
        non-rope Q depth (`padded_nope_depth`), not the V/output depth. They
        coincide for MHA / DeepSeek MLA.
        """
        return self.BM * self.padded_nope_depth * Self.qkv_dtype_size

    @always_inline
    def q_rope_bytes(self) -> Int:
        """Q rope region bytes. Uses rope_dtype_size when set, else dtype_size.
        """
        return self.BM * self.rope_depth() * Self.rope_dtype_size

    @always_inline
    def rope_depth(self) -> Int:
        """Depth of the rope part. Calculated as:
        padded_qk_depth - padded_nope_depth (0 for MHA where qk_depth ==
        nope_depth). Uses the non-rope Q/K width (`padded_nope_depth`), NOT the
        V/output depth — the two differ when `v_head_dim != qk_nope_head_dim`.
        """
        return self.padded_qk_depth - self.padded_nope_depth

    @staticmethod
    @always_inline
    def crossp_supported(
        num_q: Int,
        use_ws: Bool,
        pair_cta: Bool,
        use_shared_kv: Bool,
        rope_depth: Int,
        page_size: Int,
        BN: Int,
    ) -> Bool:
        """The cross-stage-P support matrix, in one place.

        Every `comptime assert` that guards a cross-stage-P code path derives
        its condition from here, so turning the feature on BY DEFAULT can never
        trip one of those guards. They stay hard errors, but only an explicit
        `-D FA4_TMEM_CROSS_P=true` on an unsupported config can reach them.

        Deriving the default from the guards is the point. An earlier version
        of this predicate listed the consumers it knew about instead, and CI
        found a consumer it did not know about within the hour.

        The conjuncts, each traceable to the site that requires it:
        - `num_q == 2`, `not use_ws`: the P windows only exist on the 2Q
          non-warp-specialized layout (`TMEM_P0`/`TMEM_P1` above).
        - `use_shared_kv`, `not pair_cta`: `mma_warp` asserts the shared-KV
          path; the split-KV both-lazy and pair-CTA schedules have no cross-P
          implementation.
        - `rope_depth == 0`: MHA only. MLA shares this config type but splits
          Q/K into nope + rope and has no cross-P path.
        - full pages: `load_warp`'s K-ahead producer asserts no partial pages.

        Taking the fields loose because `__init__` needs this before `self` is
        fully initialized, and Mojo forbids calling a method on a partly-built
        `self`.
        """
        return (
            num_q == 2
            and not use_ws
            and not pair_cta
            and use_shared_kv
            and rope_depth == 0
            and not (page_size > 0 and page_size < BN)
            and not EnableForcedOrdering
        )

    @always_inline
    def crossp_on(self) -> Bool:
        """Whether cross-stage P applies to THIS config.

        The single source of truth for the cross-stage-P decision. Both the
        `smem_used` mbar accounting below and `FA4MiscMBars.CrossP_enabled`
        derive from this, so the two can never disagree — a mismatch shows up
        as an `smem_used != smem_size` constraint failure, or worse, an
        ILLEGAL_ADDRESS at runtime.

        `FA4_TMEM_CROSS_P` now defaults ON, so `crossp_supported` is what keeps
        every other consumer byte-identical to cross-P-off.
        """
        return EnableTMEMCrossP and Self.crossp_supported(
            self.num_q,
            self.use_ws,
            self.pair_cta,
            self.use_shared_kv,
            self.rope_depth(),
            self.page_size,
            self.BN,
        )

    @always_inline
    def num_rope_buffers(self) -> Int:
        """Number of separate rope smem buffers (shared mode only).

        In shared mode K tiles alternate with V tiles in the pipeline.
        At most ceildiv(num_kv_stages, 2) K tiles can be in-flight
        simultaneously, so we only need that many rope buffers.
        For MHA (rope_depth=0), no rope buffers are needed.
        """
        if self.use_shared_kv and self.rope_depth() > 0:
            return ceildiv(self.num_kv_stages, 2)
        return 0

    @always_inline
    def num_k_scale_bufs(self) -> Int:
        """Number of staged k_scale smem buffers.

        In shared mode, K tiles alternate with V tiles so at most
        ceildiv(num_kv_stages, 2) K tiles are in-flight simultaneously.
        In non-shared mode, each KV stage has its own K buffer.
        Returns 0 when scale_dtype_size == 0 (no per-token scaling).
        """
        if self.scale_dtype_size == 0:
            return 0
        if self.use_shared_kv:
            return ceildiv(self.num_kv_stages, 2)
        return self.num_kv_stages

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        qk_depth: Int,
        ov_depth: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        is_mla: Bool,
        pair_cta: Bool = False,
        BM: Int = 256,
        num_qk_stages: Int = 0,
        splitk_partitions: Int = 1,
        dynamic_cluster_dim: Bool = False,
        nope_depth: Int = -1,
        single_o: Bool = False,
        bn_cap: Int = 0,
        ws_shared_key: Bool = False,
    ):
        # num_qk_stages == 0 (default) derives the optimal Q@K' staging.
        # A nonzero value pins it (used by the in-kernel 1Q/2Q switch, which
        # requires the 1Q variant's staging to match the 2Q config's — see
        # `switch_1q_config`). The caller must pass a value that is valid for
        # this shape, i.e. one the constructor could itself derive for the
        # same `padded_qk_depth`/`swizzle_mode`. If the pinned staging's extra
        # barriers do not fit in smem, the constructor falls back to 1 stage.
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.qk_depth = qk_depth
        self.pair_cta = pair_cta
        self.BM = BM
        # `BM` is the primary knob; `num_q` (Q sub-tiles per BM tile) and
        # `MMA_M` are derived from it:
        #   BM=256 -> 2Q, MMA_M=128 (single-CTA) or 256 (pair-CTA)
        #   BM=128 -> 1Q, MMA_M=128 (single-CTA)
        #   BM=64  -> 1Q, MMA_M=64  (warp-specialized packed-TMEM / Layout-E)
        #   BM=32  -> 1Q, MMA_M=32  (warp-specialized packed-TMEM / Layout-G)
        self.num_q = 2 if BM == 256 else 1
        # single_o implies num_q==1 (the body's 1Q path aliases O). Guard
        # against an inconsistent caller; `single_o=False` is the default and
        # leaves every existing config untouched.
        self.single_o = single_o and self.num_q == 1
        self.page_size = page_size
        self.is_mla = is_mla
        self.splitk_partitions = splitk_partitions
        self.dynamic_cluster_dim = dynamic_cluster_dim
        if pair_cta:
            # Pair-CTA shares one MMA across 2 CTAs (BM must be 256).
            self.MMA_M = 256
        elif BM == 64:
            # Warp-specialized packed-TMEM (1x2 / Layout-E) datapath.
            self.MMA_M = 64
        elif BM == 32:
            # Warp-specialized packed-TMEM (1x4 / Layout-G) datapath.
            self.MMA_M = 32
        else:
            self.MMA_M = 128
        self.fuse_gqa = group > 1 and (self.MMA_M % group == 0) and not is_mla
        comptime if Self.qkv_dtype.is_float8():
            self.swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.swizzle_mode = swizzle_mode
        var swizzle_elems = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        self.ov_depth = ov_depth
        # `nope_depth < 0` (default) means "no separate nope dim" — used by MHA
        # and by DeepSeek-style MLA where the non-rope Q/K width equals the V
        # head dim. In that case nope tracks ov, so every padded_nope_depth use
        # is byte-identical to the pre-decoupling padded_ov_depth.
        self.nope_depth = ov_depth if nope_depth < 0 else nope_depth
        self.padded_qk_depth = align_up(qk_depth, swizzle_elems)
        self.padded_ov_depth = align_up(ov_depth, swizzle_elems)
        self.padded_nope_depth = align_up(self.nope_depth, swizzle_elems)

        # we use two q and o
        # determine BN via tmem. The TMEM column budget (512) holds S
        # accumulators (2*BN) plus O accumulators:
        #   2-O (default):  2*BN + 2*ov <= 512 -> BN <= 256 - ov
        #   single-O:       2*BN + 1*ov <= 512 -> BN <= (512 - ov)/2
        # The KV tile must hold the nope-wide K_nope AND the v-wide V, so the O
        # term is bounded by the wider of the two (when v_head_dim < qk_nope,
        # using the smaller padded_ov alone would inflate BN and starve KV
        # stages). Byte-identical for MHA / DeepSeek (nope == ov). NB: inline
        # `max` (not `shared_kv_cols()`) — `self` is partially initialized here, so
        # a method call (which borrows all of `self`) is illegal before BN.
        # Warp-specialized packed-TMEM fires for cta_group==1 and MMA_M<=64
        # (mirrors SM100TensorAccumulator.use_ws): Layout-G (1x4, MMA_M=32,
        # m_pack=4) and Layout-E (1x2, MMA_M=64, m_pack=2). It packs `m_pack`
        # score rows onto the same physical TMEM columns, so each S/P
        # accumulator occupies BN/m_pack physical columns (m_pack=1 => no
        # packing => byte-identical to the non-WS path).
        var use_ws = (not pair_cta) and self.MMA_M <= 64
        var m_pack = 128 // self.MMA_M if use_ws else 1
        # Store the derived flags so consumers (e.g. fa4_softmax) read them from
        # the config instead of re-deriving the expression above.
        self.use_ws = use_ws
        self.m_pack = m_pack
        # Shared-key mode is only meaningful for the warp-specialized 1Q
        # datapath. Guard against an inconsistent caller the way `single_o`
        # does; `ws_shared_key=False` is the default and leaves every existing
        # config untouched.
        #
        # `not is_mla and nope == ov` is NOT redundant belt-and-braces: it is
        # what makes `_o_cols` below unambiguous. `_o_cols` serves two masters
        # -- the O accumulator's TMEM columns AND a proxy for the KV tile width
        # (which must hold the nope-wide K_nope as well as the v-wide V). Those
        # two coincide only when `nope == ov`, so the shared-key arm may
        # substitute the O footprint for the whole `max` only under that
        # premise. WS already requires `not is_mla` in `supported()`; this
        # makes the invariant local and self-guarding instead of remote.
        self.ws_shared_key = (
            ws_shared_key
            and use_ws
            and self.num_q == 1
            and not is_mla
            and self.padded_nope_depth == self.padded_ov_depth
        )
        # Physical TMEM columns charged to the O term of the BN budget. Mode-off
        # this is `max(nope, ov)` verbatim. Mode-on the accumulator is packed
        # `m_pack`-ways into depth bands, so it really occupies
        # `padded_ov / m_pack` columns -- and charging the un-divided width is
        # what drives `_bn_budget` to 0 at depth 256 and negative at 512.
        var _o_cols = Self._o_phys_cols(
            max(self.padded_nope_depth, self.padded_ov_depth),
            m_pack,
            self.ws_shared_key,
        )
        # Packing multiplies the achievable BN by m_pack: the two S accumulators
        # occupy 2*(BN/m_pack) columns instead of 2*BN, so BN may be m_pack larger.
        var _bn_budget = (
            (Self.sm100_tmem_cols - _o_cols)
            // 2 if self.single_o else Self.sm100_tmem_cols
            // 2
            - _o_cols
        ) * m_pack
        self.BN = min(256, align_down(_bn_budget, Self.MMA_K))
        # `bn_cap > 0` clamps BN below the TMEM-max so the SMEM budget can fit
        # >= 2 KV stages. Only the single-O wide-V fallback passes a cap; the
        # default (bn_cap == 0) leaves every existing BN untouched.
        if bn_cap > 0:
            self.BN = min(self.BN, align_down(bn_cap, Self.MMA_K))
        # page_size == 0 means non-paged (no constraint).
        # page_size >= BN: page contains full tile (page_size % BN == 0).
        # page_size < BN: tile spans multiple pages (BN % page_size == 0).
        if (
            page_size != 0
            and page_size % self.BN != 0
            and self.BN % page_size != 0
        ):
            self.BN = prev_power_of_two(self.BN)
        # Row-major (page-dense) K/V is the default in the multi-page paging
        # regime (0 < page_size < BN); single-page / continuous / ragged
        # (page_size == 0 or >= BN) stay chunk-outer. MLA is structurally
        # chunk-outer (its own MMA warps never fold), so it is excluded here.
        # v_row_major()/k_row_major() still apply the full
        # geometry/swizzle/pair-CTA/_kv_fold_base_ok feasibility gating.
        var page_dense_default = (
            not is_mla and page_size > 0 and page_size < self.BN
        )
        self.row_major_v_atoms = page_dense_default
        self.row_major_k_atoms = page_dense_default
        # S/P score accumulators occupy BN/m_pack physical TMEM columns under
        # the packed WS datapath (m_pack=1 => full BN, byte-identical). The O
        # term used to be unconditionally `padded_ov`; it is now
        # `_o_phys_cols` below, which divides by `m_pack` under shared-key.
        var s_cols = self.BN // m_pack
        self.TMEM_S1 = Self.TMEM_S0 + s_cols
        # Cross-stage: P0 in S1's window, P1 in S0's window, at the region
        # base. bf16 P needs 64 of the 128 f32 columns; any 32-col-aligned
        # in-region offset satisfies the tcgen05 A-operand (FlashInfer uses
        # +32 only to clear row stats it keeps inplaced in TMEM — MAX keeps
        # stats in SMEM, so the base is free). 2Q + non-WS only: the 1Q
        # odd-T tail aliases s1->s0, which would collide the two P windows.
        comptime if EnableTMEMCrossP:
            if self.num_q == 2 and not self.use_ws:
                self.TMEM_P0 = self.TMEM_S1
                self.TMEM_P1 = Self.TMEM_S0
            else:
                self.TMEM_P0 = Self.TMEM_S0
                self.TMEM_P1 = self.TMEM_S1
        else:
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_P1 = self.TMEM_S1
        self.TMEM_O0 = self.TMEM_S1 + s_cols
        # Physical TMEM columns one O accumulator occupies. `_o_phys_cols` is a
        # `@staticmethod` precisely so this call is legal here (a method would
        # borrow the partially initialized `self`) -- so the constructor and
        # `o_phys_cols()` are ONE spelling, not two kept in sync by hand.
        var o_phys = Self._o_phys_cols(
            self.padded_ov_depth, m_pack, self.ws_shared_key
        )
        # single-O: alias O1 onto O0 (the 1Q body reuses one O accumulator) and
        # reserve a single O region -> tmem_used = 2*BN + padded_ov. Default
        # (2-O) is unchanged: two distinct O regions, tmem_used = 2*BN + 2*ov.
        if self.single_o:
            self.TMEM_O1 = self.TMEM_O0
        else:
            self.TMEM_O1 = self.TMEM_O0 + o_phys
        self.tmem_used = self.TMEM_O1 + o_phys

        # We have the following resources that need smem barriers:
        # KV: num_kv_stages
        # S: 2
        # C: 2
        # O: 2
        # softmax order: 2
        # q: 1, for Q1 synchronization
        # 4 for `o_pipeline` (2 consumer + 2 producer)
        # we need two per stage
        # Compute staging for Q@K' and P@V operations
        # num_qk_stages: Controls how K loading is pipelined for Q@K' MMA
        # num_pv_stages: Controls how P writing is pipelined for P@V MMA
        #
        # For Q@K': K can be loaded in stages, MMA starts after first stage arrives
        # For P@V: V must be complete, but P writing can be staged to unblock MMA sooner
        #
        # Divisibility constraints:
        # - num_qk_stages must divide padded_depth (for K column splitting)
        # - num_pv_stages must divide BN (for P column splitting)
        # - Both must respect MMA_K alignment (16 elements)
        #
        # Staging infrastructure:
        # - SM100TensorAccumulator.mma (both a_tmem=False/True quadrants)
        #   supports a stage_idx parameter for processing in chunks when
        #   num_stages > 1
        # - KPipeline and VPipeline structs support separate K/V barrier management
        # - FA4MiscMBars is parameterized by num_pv_stages for S barriers
        # - load() loads K in num_qk_stages chunks with separate barriers per stage
        # - store_exp() writes P in num_pv_stages chunks with barriers per stage
        # - mma() loops over qk_stages for Q@K' and pv_stages for P@V
        #
        # Computed staging values:
        # - num_qk_stages: How many chunks to split K processing into for Q@K' MMA
        # - num_pv_stages: How many chunks to split P writing into for P@V MMA
        #
        if is_mla:
            self.num_qk_stages = 1
        elif num_qk_stages != 0:
            self.num_qk_stages = num_qk_stages
        else:
            # Q@K' staging is enabled: MMA processes K in num_qk_stages chunks,
            # allowing register pressure reduction and potential overlap.
            self.num_qk_stages = gcd(
                self.padded_qk_depth // swizzle_elems,
                self.padded_qk_depth // Self.MMA_K,
            )

        # P@V staging requires coordinated changes to store_exp and mma functions:
        # - store_exp must write P in stages and signal barriers per stage
        # - mma must wait for each P stage barrier before processing
        self.num_pv_stages = 2

        var smem_use = 4
        # Compute misc_mbars fixed size (barriers that don't scale with num_kv_stages):
        # - S consumers: 2 * num_pv_stages (num_pv_stages per warp group)
        # - S producers: 2 (1 per warp group)
        # - C barriers: 4 (C0/C1 producer/consumer)
        # - Order barriers: 2 (only when EnableForcedOrdering)
        # - Q1Sync barriers: num_qk_stages (only when num_q == 2; num_q=1
        #   shares Q across both pipelines so no Q1Sync slot is needed —
        #   FA4MiscMBars collapses Q1SyncIdx in that mode)
        # - O producers: 2 (O consumers reuse S_consumer[0], not separate)
        # - Split-K publish barrier: 1 (only when num_q == 1 and
        #   splitk_partitions > 1; matches FA4MiscMBars.Publish_count and
        #   FA4Config.splitk_dynamic()). Without it the split-K launch reserves
        #   `smem_used` 8 bytes short of the SM100AttentionSMem layout, and the
        #   publish-barrier init writes out of bounds (KERN-3172).
        # Total fixed = 8 + order_barrier_count + 2*num_pv_stages
        #             + (num_qk_stages if num_q == 2 else 0)
        #             + (1 if num_q == 1 and splitk_partitions > 1 else 0)
        comptime order_barrier_count: Int = 2 if EnableForcedOrdering else 0
        # Cross-stage P appends 10 mbars (2 sfree + 2x4 depth-4 inplace).
        # Cross-stage P's 10 mbars are added further down, once
        # `use_shared_kv` is decided -- `crossp_supported` needs it.
        var misc_mbars_fixed_size = (
            8
            + order_barrier_count
            + 2 * self.num_pv_stages
            + (self.num_qk_stages if self.num_q == 2 else 0)
            + (1 if self.num_q == 1 and self.splitk_partitions > 1 else 0)
        )
        smem_use += misc_mbars_fixed_size * Self.mbar_size

        # rope occupies the Q/K columns past the non-rope (nope) part, so it is
        # padded_qk - padded_nope (NOT padded_ov, which is the V/output depth).
        var rope_depth = self.padded_qk_depth - self.padded_nope_depth

        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        # Q region: when rope_dtype_size > 0, Q nope and Q rope have different
        # dtype sizes (e.g. FP8 nope + BF16 rope for per-token-scale MLA). The
        # Q_nope sub-region is `padded_nope_depth` wide (the Q@K_nope' width).
        var qk_depth_bytes: Int
        comptime if Self.rope_dtype_size > 0:
            qk_depth_bytes = (
                self.padded_nope_depth * Self.qkv_dtype_size
                + rope_depth * Self.rope_dtype_size
            )
        else:
            qk_depth_bytes = self.padded_qk_depth * Self.qkv_dtype_size
        smem_use += self.BM * qk_depth_bytes
        # q_scale: always 1 buffer (per-token scale only; 0 when no scaling).
        smem_use += self.BM * Self.scale_dtype_size
        # Add space for correction smem when not using tmem for correction.
        # Must match `SM100AttentionSMem.correction_bytes` in smem.mojo: the
        # layout reserves one Float32 slot per softmax thread, i.e.
        # `2 * WARPGROUP_SIZE = 256` Float32 entries (1 KiB) regardless of
        # `num_q` or `BM` (the correction store is indexed by CTA-wide `tid`,
        # not by `BM`). A `BM`-derived size only happens to be right at
        # BM==128/256; for the warp-specialized MMA_M=32 path (BM=32) it would
        # under-reserve and the trailing mbar / tmem_addr regions overflow
        # into unmapped __shared__ on init.
        smem_use += (
            2
            * WARPGROUP_SIZE
            * Self.num_correction_cols
            * size_of[DType.float32]()
        )

        # Shared-key regions. BOTH fold to 0 for every config that ships
        # today, so `smem_used` and every derived `num_kv_stages` are
        # byte-identical off-mode.
        #
        # These MUST be charged HERE, before `var remaining = ...` below:
        # `remaining` is the numerator of `kv_slots = remaining //
        # bytes_per_subtile`, so bytes added after it come out of nothing and
        # over-subscribe the ring. The `smem_used >= smem_size()` check in the
        # kernel is one-directional and would NOT catch that -- it only
        # catches under-reservation. Charging them here instead costs the ring
        # `ceildiv(bytes, bytes_per_subtile)` slots honestly, which
        # `supported()`'s `num_kv_stages >= ring_slots_needed()` then judges.
        smem_use += Self._p_smem_bytes(self.BM, self.BN, self.ws_shared_key)
        smem_use += Self._ws_exchange_bytes(self.ws_shared_key)

        # We use one of two strategies:
        #  - non-shared kv: more efficient/neater to track smem separately.
        #              nope and rope smem can be tracked together
        #  - shared kv: if the maximum number of `nope`s we can store is odd
        #              then splitting into two rings would require us to round
        #              down to an even number of stages. Sharing avoids this.
        # We divide bytes needed by `k` and `v` into shared and k-specific:
        # In pair-CTA mode each CTA stores half of K/V:
        # K: BN/2 rows × full depth, V: full BN rows × ov_depth/2 cols.
        # The shared K_nope/V buffer stage fits the wider of K_nope/V; pair-CTA
        # halves it below. Inline `max` (not `shared_kv_cols()`) — `self` is
        # partially initialized here (a method call would borrow all of `self`).
        var kv_data_elems = self.BN * max(
            self.padded_nope_depth, self.padded_ov_depth
        )
        if pair_cta:
            kv_data_elems //= 2
        var bytes_per_kv = (
            kv_data_elems * Self.qkv_dtype_size + 2 * Self.mbar_size
        )  # KV barriers
        var kv_rows = self.BN // 2 if pair_cta else self.BN
        var bytes_per_k = (
            kv_rows * rope_depth * Self.rope_dtype_size
            + kv_rows * Self.scale_dtype_size
        )  # k scale buffers

        # total k + v bytes is thus
        #   kv_slots * bytes_per_kv + ceildiv(kv_slots, 2) * bytes_per_k
        # If `kv_slots` is even we use the non-shared pipelines (dedicated K and
        # V rings); if odd, the shared ring (K and V (sub-)tiles interleaved).

        var remaining = Self.sm100_smem_carveout - smem_use
        if use_ws:
            # WS depth-split KV: K and V are BOTH split by depth into uniform
            # BN x (depth // num_qk_stages) sub-tiles, so one ring can hold
            # either and the shared path works even with num_qk_stages > 1.
            # Count the sub-tile slots that fit, then pick the pipeline by
            # parity. rope/scale are 0 on the WS MHA path (enforced by
            # supported()), so a sub-tile is purely K/V data + its 2 barriers.
            var sub_depth = (
                max(self.padded_nope_depth, self.padded_ov_depth)
                // self.num_qk_stages
            )
            var bytes_per_subtile = (
                self.BN * sub_depth * Self.qkv_dtype_size + 2 * Self.mbar_size
            )
            var kv_slots = remaining // bytes_per_subtile
            smem_use += kv_slots * bytes_per_subtile
            # WS always uses the SHARED sub-tile ring: K depth-halves and V
            # depth-tiles interleave in ONE ring of `kv_slots` 32768-B slots, so
            # the pipeline stride (one sub-tile per slot) matches this
            # reservation by construction. (The non-shared parity split reserved
            # sub-tiles but the pipeline strided full 65536-B tiles -> 2x
            # overrun; the shared ring removes that impedance mismatch.)
            # `num_qk_stages` stays as derived (2 at depth=128); NOT forced to 1
            # — it defines the sub-tile depth (ws_subtile_bytes / BK0). The depth
            # split is expressed as the slot SEQUENCE (2 K slots + 2 V slots per
            # block), not as an intra-slot stride.
            self.use_shared_kv = True
            self.num_kv_stages = kv_slots
        else:
            # remaining >= kv_slots * bytes_per_kv
            #   + ceildiv(kv_slots,2) * bytes_per_k
            #   = kv_slots * (bytes_per_kv + bytes_per_k/2) (kv_slots even)
            var kv_slots = remaining // (bytes_per_kv + bytes_per_k // 2)
            # A pinned num_qk_stages > 1 requires the non-shared pipeline (the
            # shared ring never stages K), so round an odd slot count down to
            # even to force the non-shared path below.
            if num_qk_stages > 1 and kv_slots % 2 == 1:
                kv_slots -= 1
            var bytes_used = (
                kv_slots * bytes_per_kv + ceildiv(kv_slots, 2) * bytes_per_k
            )
            if bytes_used > remaining:
                kv_slots -= 1
                bytes_used = (
                    kv_slots * bytes_per_kv + ceildiv(kv_slots, 2) * bytes_per_k
                )
            smem_use += bytes_used

            # single-O (1Q wide-V) always uses the non-shared pipeline (separate
            # K and V), never the shared ring. The single-O serial P@V path (one
            # warp group folds every K/V tile into the aliased O0) is validated
            # only on the non-shared pipeline; the shared ring interleaves K/V in
            # the even/odd pair order, which the single-O per-tile consumption
            # does not match. Forcing non-shared keeps ONE single-O code path.
            # `supported()` (>= 2 KV stages) then rejects any wide-V shape that
            # cannot afford non-shared staging, at compile time. Non-single-O
            # configs are unaffected (byte-identical).
            if kv_slots % 2 == 1 and not self.single_o:  # odd -> shared
                self.use_shared_kv = True
                self.num_kv_stages = kv_slots
                self.num_qk_stages = 1
            else:
                self.use_shared_kv = False
                self.num_kv_stages = kv_slots // 2
                if is_mla:
                    self.num_qk_stages = 1
                else:
                    # we try to split num_qk_stages
                    if num_qk_stages != 0:
                        self.num_qk_stages = num_qk_stages
                    else:
                        self.num_qk_stages = gcd(
                            self.padded_qk_depth // swizzle_elems,
                            self.padded_qk_depth // Self.MMA_K,
                        )
                    # we need an extra bytes
                    var barrier_bytes_per_stage = (
                        self.num_kv_stages * 2 * Self.mbar_size
                    )
                    var total_smem_use = (
                        smem_use
                        + (self.num_qk_stages - 1) * barrier_bytes_per_stage
                    )
                    if total_smem_use < Self.sm100_smem_carveout:
                        smem_use = total_smem_use
                    else:
                        self.num_qk_stages = 1

        # BK0: K-dimension chunk size for Q@K' per stage
        self.BK0 = self.padded_qk_depth // self.num_qk_stages
        # BK1: Full BN since V loading is not staged (V must be complete
        # for P@V)
        self.BK1 = self.BN

        # Cross-stage P appends 10 mbars (2 sfree + 2x4 depth-4 inplace). Added
        # here rather than with the other misc mbars because `crossp_supported`
        # needs `use_shared_kv`, which is only decided above. Same predicate
        # FA4MiscMBars.CrossP_count is threaded from, so the accounting and the
        # layout cannot drift.
        if EnableTMEMCrossP and Self.crossp_supported(
            self.num_q,
            self.use_ws,
            self.pair_cta,
            self.use_shared_kv,
            self.padded_qk_depth - self.padded_nope_depth,
            self.page_size,
            self.BN,
        ):
            smem_use += 10 * Self.mbar_size

        # BLASST skip-vote region: must match SM100AttentionSMem.blasst_vote_bytes
        # or the launch smem_used undershoots -> CUDA_ERROR_ILLEGAL_ADDRESS.
        comptime if get_defined_bool["ENABLE_BLASST", False]():
            smem_use += 2 * 2 * 4 * size_of[UInt8]()

        self.smem_used = smem_use

    def supported(self) -> Bool:
        # Runtime-k partial-page contraction (mma_maybe_partial_k, used only
        # by the non-MLA fa4_mma path) cuts the P@V contraction at the loaded
        # V boundary to avoid reading uninitialized SMEM. That cut is only
        # safe when the loaded region is MMA_K-aligned, i.e. page_size is a
        # multiple of MMA_K. A sub-tile page (page_size < BN) that is not
        # MMA_K-aligned is therefore unsupported here. MLA prefill has its own
        # MMA warps (does not use fa4_mma) and is exempt.
        if (
            not self.is_mla
            and self.page_size != 0
            and self.page_size < self.BN
            and self.page_size % Self.MMA_K != 0
        ):
            return False
        # Split-K (cluster partitioning of K/V) is only wired for the
        # num_q==1 single-CTA path; any other config must leave it disabled.
        if self.num_q != 1 and self.splitk_partitions != 1:
            return False
        var base = (
            self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
            # BM is the primary knob; only 32/64 (WS), 128, 256 are valid tiles.
            and (
                self.BM == 32
                or self.BM == 64
                or self.BM == 128
                or self.BM == 256
            )
            # The warp-specialized datapath (BM=32 Layout-G, BM=64 Layout-E) is
            # single-CTA, MHA-only, and its KV budget assumes no rope/scale
            # sub-tile bytes. `not use_ws` (BM in {128,256}) skips this block.
            and (
                not self.use_ws
                or (
                    not self.pair_cta
                    and not self.is_mla
                    and self.rope_depth() == 0
                    and Self.scale_dtype_size == 0
                    # WS shared sub-tile ring, from the schedule's own span
                    # model rather than a literal. Load-bearing on the
                    # shared-key arm: an under-tight floor admits `fp8 d512`
                    # with 11 slots against a true floor of 16, which HANGS
                    # rather than answering wrongly.
                    and self.num_kv_stages >= self.ring_slots_needed()
                    # The SECOND floor on the same number, usually the larger:
                    # `num_kv_stages` also sizes the SMEM arena the combine
                    # epilogue carves from the dead Q+KV span. As a conjunct an
                    # inadmissible config REROUTES to the 1Q/split-K carve; the
                    # `comptime assert` in `fa4_softmax` alone would make it a
                    # build error. Layout-E d128 sits exactly on the floor.
                    and self.num_kv_stages >= self.ws_arena_slots_needed()
                    # Uniform shared-ring sub-tile premise: a ring slot holds
                    # EITHER a K depth-half OR a V sub-tile, so the two must be
                    # byte-equal. K is always a [BN x BK0] depth-half. V differs
                    # by layout:
                    #   Layout-G (m_pack=4): V is DEPTH-scattered into tiles of
                    #     width depth_tile = 256//m_pack; byte-equal to K iff
                    #     depth_tile == BK0 (both 64 at depth 64 and 128).
                    #   Layout-E (m_pack=2): V is REDUCTION(key)-split into
                    #     num_qk_stages tiles; each has (BN//m_pack)//num_qk_stages
                    #     keys x (m_pack*ov) depth cols. Byte-equality with the
                    #     [BN x BK0] K sub-tile is algebraic once the key axis
                    #     divides evenly: (BN//m_pack//nqs)*(m_pack*D)
                    #     == BN*(D//nqs) == the K sub-tile bytes.
                    #   Shared-key: V is neither depth-scattered per warp nor
                    #     reduction-split per partition -- the warpgroup walks
                    #     ONE key band and the quarters are depth bands, so a V
                    #     sub-tile is just [BN x sub_depth], byte-equal to the K
                    #     sub-tile by construction. What must hold instead is
                    #     that an output depth tile (`pv_mma_n()` wide) is a
                    #     whole number of V sub-tiles, and that one MMA's key
                    #     chunk is a whole number of P sub-stage k-blocks --
                    #     under-satisfying the latter makes `k_batch_end`'s
                    #     `min` silently DROP the last k-block rather than fail.
                    # Reject shapes that break this uniform-sub-tile premise.
                    and (
                        (
                            self.padded_ov_depth % self.kv_sub_depth() == 0
                            and self.pv_mma_n() % self.kv_sub_depth() == 0
                            and self.pv_key_chunk()
                            % (Self.MMA_K * self.num_pv_stages)
                            == 0
                        ) if self.ws_shared_key else (
                            (
                                self.m_pack == 4
                                and (256 // self.m_pack) == self.BK0
                            )
                            or (
                                self.m_pack == 2
                                and (self.BN // self.m_pack)
                                % self.num_qk_stages
                                == 0
                            )
                        )
                    )
                    # Two shared-key premises that are otherwise enforced only
                    # by a `comptime assert` deep inside a warp -- i.e. as a
                    # BUILD FAILURE on a config the dispatcher already accepted,
                    # rather than as a route rejection. Both are vacuously true
                    # off-mode, so `not ws_shared_key` short-circuits first and
                    # no shipping config re-evaluates them.
                    #
                    # (1) `fa4_ws_exchange4`'s slot map is a bijection from the
                    #     warpgroup's 128 threads onto 128 f32 slots only when
                    #     `m_pack * BM == WARPGROUP_SIZE` (thread `t` owns row
                    #     `t % BM` of warp `t / BM`). Violate it and two warps
                    #     alias one slot: a plausible-but-wrong row max, no
                    #     fault. It must stay guarded -- BM=256 breaks it, and
                    #     BM=256 is a shipping config.
                    # (2) `correction_warp._rescale_o`'s software-pipelined
                    #     prologue asserts `load_iters > 1`, where `load_iters
                    #     = cols // (2 * batch_size)` and `batch_size` is 16 for
                    #     any 16-divisible width. `cols >= 64 and cols % 32 ==
                    #     0` is a SUFFICIENT (not identical) condition for that
                    #     ladder -- deliberately simpler than reproducing it,
                    #     since a second copy of the derivation is exactly the
                    #     drift this guard exists to prevent. The warp's own
                    #     asserts remain the ground truth: if this guard is ever
                    #     weakened past them, they fire loudly at compile time.
                    #     In practice it rejects shared-key below depth 256,
                    #     which is the only depth range the mode exists for.
                    #
                    # (3) Producer/consumer must walk the SAME number of ring
                    #     positions per V tile, or the shared cursor desyncs and
                    #     the CTA hangs. The producer's `_emit_v` trip count is
                    #     `num_qk_stages`; the consumer's key-chunked walk is
                    #     `num_o_tiles()` depth tiles x `BN // pv_key_chunk()`
                    #     key chunks. `v_ring_positions_per_tile()` is that
                    #     product, and its docstring shows the two are equal by
                    #     algebra for every shared-key geometry.
                    #
                    #     Asserting the product -- not the older
                    #     `num_o_tiles() == num_qk_stages` -- is the point. That
                    #     older form is the same statement ONLY when there is
                    #     exactly one key chunk; at BN=256 it is plain false (1
                    #     vs 4 at d256, 2 vs 8 at d512) and it was serving as a
                    #     blanket refusal while the walk did not exist. Now that
                    #     it does, the honest conjunct is the frame equality.
                    #
                    #     `BN % pv_key_chunk() == 0` is not decoration: a ragged
                    #     final chunk would make the two sides' `c * KC`
                    #     arithmetic disagree with `BN`, which is a silent
                    #     wrong-answer rather than a hang.
                    #
                    # (4) FUSED cross-CTA split-K is not merely unwired under the
                    #     mode, it is BROKEN. Its two arms stage the per-CTA
                    #     (M, L) in `ws_maxsum1` and the bands in `ws_l2_stage`,
                    #     and those are `2*WS_STAGE + WS_ML` and
                    #     `2*WS_STAGE + 2*WS_ML` off the same base -- which
                    #     COLLAPSE onto one pointer when `ws_epilogue_ml_f32()`
                    #     is 0, as it is here. Rejecting is the honest answer
                    #     until someone re-carves them (there is room; nobody has
                    #     needed it). Note this does NOT touch the UNFUSED
                    #     (workspace) split-K egress -- `ws_o_row_off` plus
                    #     `_ws_write_lse` ride the plain arm, and that is the one
                    #     production decode actually routes through.
                    # (5) The shared-key epilogue walks `num_o_tiles()` windows of
                    #     `pv_mma_n() * BM` f32 inside `ws_l2_stage`, summing to
                    #     `padded_ov_depth * BM`; the carve provides
                    #     `ov_depth * BM` (`ws_epilogue_f32_slots()`). Those are
                    #     equal only when the swizzle does not pad, and nothing
                    #     above forces that -- the constructor guard requires
                    #     `not is_mla` and `padded_nope == padded_ov`, neither of
                    #     which implies `padded_ov == ov_depth`. A padded shape
                    #     would run the last window over `ws_l2_maxsum` and the
                    #     output tile, silently. Rejecting beats widening the
                    #     carve: widening moves `ws_epilogue_f32_slots()` ->
                    #     `ws_arena_slots_needed()` -> possibly `num_kv_stages`,
                    #     and it buys nothing, since every depth the mode admits
                    #     (256, 512) is already a multiple of `swizzle_elems`.
                    and (
                        not self.ws_shared_key
                        or (
                            self.m_pack * self.BM == WARPGROUP_SIZE
                            and self.correction_o_cols() >= 64
                            and self.correction_o_cols() % 32 == 0
                            and self.BN % self.pv_key_chunk() == 0
                            and self.v_ring_positions_per_tile()
                            == self.num_qk_stages
                            and self.splitk_partitions == 1
                            and self.padded_ov_depth == self.ov_depth
                        )
                    )
                )
            )
        )
        if self.num_q == 1:
            # num_q=1 is single-CTA only (pair-CTA only requires double
            # the seq-len of single-CTA, num_q=1 is for small seq-len).
            # pair-CTA decreases perf in every benchmark I've tried
            # anyway, so it especially doesn't make sense for small
            # seq-len.
            return (
                base
                and self.qk_depth >= 64
                # Shared-key mode exists precisely to reach 256/512: it shrinks
                # the per-WG O accumulator by `m_pack`, which is what keeps
                # `tmem_used` at 256/384 instead of overflowing. The 256 cap
                # stays for every other 1Q config, whose O is un-packed.
                and self.qk_depth <= (512 if self.ws_shared_key else 256)
                and not self.pair_cta
                # Split-K cluster size P. P need NOT be a power of two: the
                # scheduler tile recovery (block_idx.x // P) and the depth-band
                # split both take P as a comptime constant, so a non-pow2 P
                # lowers to a multiply-shift rather than a real divide -- the
                # combine / splitk_window math is P-general (see
                # attention_utils.splitk_window and softmax_warp's
                # reduce-scatter band split). P MUST be even, though: the
                # SIMD-2 weight-normalize loop in fa4_splitk_combine_write
                # strides by 2 (`range(0, P, 2)`), so an odd P would read
                # w[P] out of bounds. 6 and 10 fill the occupancy gaps between
                # the pow2 rungs (P=6 -> 132 SMs like P=4; P=10 -> 110 SMs).
                # P in {10, 16} exceeds the portable cluster cap (8) and is
                # non-portable, but the runtime sets
                # NON_PORTABLE_CLUSTER_SIZE_ALLOWED on every function load
                # (CUDADeviceContext::loadFunction), so those clusters are
                # launchable on B200 without extra plumbing.
                and (
                    self.splitk_partitions == 1
                    or self.splitk_partitions == 2
                    or self.splitk_partitions == 4
                    or self.splitk_partitions == 6
                    or self.splitk_partitions == 8
                    or self.splitk_partitions == 10
                    or self.splitk_partitions == 16
                )
            )
        if self.pair_cta:
            # Pair-CTA: depth > 64 (depth=64 needs 32B swizzles) and <= 128.
            return base and self.qk_depth > 64 and self.qk_depth <= 128
        return base and self.qk_depth >= 64

    @always_inline
    def with_num_q(self, num_q: Int, *, num_qk_stages: Int = 0) -> Self:
        """Reconstruct this config with a different `num_q` (single-CTA).

        `num_qk_stages == 0` (default) lets the constructor derive the
        optimal staging for the new shape — appropriate for the dispatch-time
        1Q/2Q selection, where each launch config is free-standing. A nonzero
        value pins the staging (see `switch_1q_config`).

        `pair_cta` is forced False because `num_q == 1` is single-CTA only
        (see `supported()`). Re-passing the stored `swizzle_mode` is faithful:
        the constructor re-derives it (FP8 re-forces 64B), and it is already
        the post-override value here. The `row_major_{v,k}_atoms` fields are
        not re-passed: the constructor recomputes them from
        `page_size`/`BN`/`is_mla`, and `BN` is `num_q`-independent, so the
        value is identical to `self`'s.
        """
        return Self(
            num_q_heads=self.num_q_heads,
            group=self.group,
            qk_depth=self.qk_depth,
            ov_depth=self.ov_depth,
            swizzle_mode=self.swizzle_mode,
            page_size=self.page_size,
            is_mla=self.is_mla,
            pair_cta=False,
            # `num_q` maps to BM: 2Q -> BM=256, 1Q -> BM=128 (single-CTA,
            # MMA_M=128). Callers only ever request num_q==1 on non-WS/MLA
            # configs, so BM=128 is the faithful 1Q reconstruction.
            BM=256 if num_q == 2 else 128,
            num_qk_stages=num_qk_stages,
            dynamic_cluster_dim=self.dynamic_cluster_dim,
            nope_depth=self.nope_depth,
            # Preserve single-O only when the reconstructed config is itself 1Q.
            # The existing prefer_1q short-seq path calls with_num_q(1) on a
            # single_o=False 2Q config -> stays single_o=False (byte-identical).
            single_o=self.single_o and num_q == 1,
            ws_shared_key=self.ws_shared_key,
        )

    @always_inline
    def with_splitk(self, splitk_partitions: Int) -> Self:
        """Reconstruct this config with a split-K cluster size (num_q==1).

        Split-K groups `splitk_partitions` single-CTA kernels in a launch
        cluster that partition the K/V sequence and (from M4) combine via
        DSMEM. `pair_cta` is forced False — split-K is single-CTA only: each
        CTA runs its own `cta_group::1` MMA over its own TMEM/SMEM, and the
        cluster exists purely to group the split-K partitions. `num_q` and the
        derived `num_qk_stages` are preserved, so `with_splitk(1)` is a no-op
        (identical config) and the split-K plumbing folds away.

        `nope_depth` (the Q@K'/Q_nope width) and `single_o` (the wide-V 1Q TMEM
        mode) are re-passed so a GLM-style config (`v_head_dim != qk_nope`) or a
        single-O config survives the reconstruction; both are byte-identical for
        the DeepSeek/MHA shapes (nope == ov, single_o == False).
        """
        return Self(
            num_q_heads=self.num_q_heads,
            group=self.group,
            qk_depth=self.qk_depth,
            ov_depth=self.ov_depth,
            swizzle_mode=self.swizzle_mode,
            page_size=self.page_size,
            is_mla=self.is_mla,
            pair_cta=False,
            # Preserve the full shape via BM (incl. WS BM=32 for the split-K
            # composition); pair_cta forced False makes BM=256 -> MMA_M=128.
            BM=self.BM,
            num_qk_stages=self.num_qk_stages,
            splitk_partitions=splitk_partitions,
            dynamic_cluster_dim=self.dynamic_cluster_dim,
            nope_depth=self.nope_depth,
            single_o=self.single_o,
            ws_shared_key=self.ws_shared_key,
        )

    @always_inline
    def with_bm(self, bm: Int, *, ws_shared_key: Bool = False) -> Self:
        """Reconstruct this config with an explicit `BM` (single-CTA).

        Used by dispatch to force a warp-specialized packed-TMEM datapath
        (`BM=32` -> `MMA_M=32`, `m_pack=4`, Layout-G; `BM=64` -> `MMA_M=64`,
        `m_pack=2`, Layout-E), with `use_ws=True`, for short prompts. `pair_cta`
        is forced False (BM=32/64 are single-CTA only, per `supported()`).
        `num_qk_stages`/`splitk_partitions`/`single_o` are left at their
        constructor defaults (derive staging, no split-K, 2-O) so the
        reconstruction is byte-identical to a direct `FA4Config(..., BM=bm)`
        build; `use_ws`/`m_pack` are derived from the new `BM`. `nope_depth` is
        re-passed so a GLM-style shape survives (byte-identical for MHA where
        nope == ov).

        `ws_shared_key` is an explicit parameter rather than inherited: this
        method deliberately resets the derived knobs, and shared-key mode is
        *selected* here (a `BM=32`/`BM=64` reconstruction is exactly where it
        becomes legal). It defaults to False, so every existing call is
        unchanged.
        """
        return Self(
            num_q_heads=self.num_q_heads,
            group=self.group,
            qk_depth=self.qk_depth,
            ov_depth=self.ov_depth,
            swizzle_mode=self.swizzle_mode,
            page_size=self.page_size,
            is_mla=self.is_mla,
            pair_cta=False,
            BM=bm,
            nope_depth=self.nope_depth,
            ws_shared_key=ws_shared_key,
        )

    @always_inline
    def ws_shared_key_vehicle(self) -> Self:
        """The shared-key config the deep (d256/d512) decode route instantiates.

        A named accessor rather than an expression spelled at each site,
        because one site is a *check on the other*:
        `test_fa4_config_scaffold`'s `_check_deep_target` asserts `supported()`
        on this spelling while dispatch instantiates it. Spelled twice, the
        check could drift off the thing checked and keep passing -- and the
        dispatch guard's false arm is SILENT, so the failure mode is a green
        build that compiled no shared-key body at all.
        """
        return self.with_bm(32, ws_shared_key=True)

    @always_inline
    def switch_1q_config(self) -> Self:
        """The 1Q variant used by the in-kernel per-sequence 1Q/2Q switch.

        Unlike the dispatch-time conversion (`with_num_q(1)`), which is free
        to pick the optimal staging, this pins `num_qk_stages` to this (2Q)
        config's value: the switch feeds the 2Q-built TMA ops to the 1Q body,
        so the per-stage K split (`QTMATile`'s smem-tile last dim and
        `k_tma`'s `BK = padded_qk_depth // num_qk_stages`) must match. The
        pinned value is always arithmetically valid here because
        `padded_qk_depth` and `swizzle_mode` are identical across the two
        configs; if its extra barriers do not fit in 1Q smem, the constructor
        falls back to 1 stage and `can_switch_to_1q()` rejects the switch.
        """
        return self.with_num_q(1, num_qk_stages=self.num_qk_stages)

    @always_inline
    def can_switch_to_1q(self) -> Bool:
        """Whether a 2Q-launched kernel may dispatch to the 1Q body at runtime.

        True only when this is a 2Q single-CTA config AND a valid 1Q variant
        exists whose TMA-op types match the 2Q ones. `switch_1q_config()`
        pins `num_qk_stages` (the one TMA-op parameter that could otherwise
        diverge — `BN`, the per-half Q `BM` (128), `v_tma_op`, and
        `ragged_tma_store` already match), so the equality check below only
        fails when the pinned staging could not be honored (smem fallback to
        1 stage). When this returns False the kernel runs pure 2Q.
        """
        if self.num_q != 2 or self.pair_cta:
            return False
        var cfg1 = self.switch_1q_config()
        return cfg1.supported() and cfg1.num_qk_stages == self.num_qk_stages

    @always_inline
    def launch_smem_used(self) -> Int:
        """Dynamic smem to reserve when launching this config's kernel.

        When the launched kernel may dispatch to the 1Q body at runtime
        (`can_switch_to_1q()`), it constructs the 1Q `SM100AttentionSMem` over
        the same dynamic smem region, so the launch must reserve the max of
        both footprints. Otherwise this is just `smem_used`.
        """
        if self.can_switch_to_1q():
            return max(self.smem_used, self.switch_1q_config().smem_used)
        return self.smem_used

    def description(self) -> String:
        return String(
            "pair_cta = ",
            self.pair_cta,
            "\nnum_q = ",
            self.num_q,
            "\nBM = ",
            self.BM,
            "\nMMA_M = ",
            self.MMA_M,
            "\nqk_depth = ",
            self.qk_depth,
            "\nBN = ",
            self.BN,
            "\nnum_kv_stages = ",
            self.num_kv_stages,
            "\ntmem_used = ",
            self.tmem_used,
            "\nsmem_used = ",
            self.smem_used,
            "\nsm100_smem_carveout = ",
            Self.sm100_smem_carveout,
            "\nnope_dtype_size = ",
            Self.qkv_dtype_size,
            "\nrope_dtype_size = ",
            Self.rope_dtype_size,
            "\nscale_dtype_size = ",
            Self.scale_dtype_size,
            "\nuse_shared_kv = ",
            self.use_shared_kv,
        )

    def correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

    def num_active_warps_per_group(self) -> Int:
        return 4

    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()
