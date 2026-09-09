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
"""MlaPrefillV2: fresh, from-scratch port of the reference MLA-prefill
INTEGRATED inner-loop architecture for AMD MI355X (gfx950).

This is a NEW kernel struct (sibling of `MlaPrefillV2Core`, not a retrofit)
that lays out the reference `mla_pfl_qh192_vh128_m32x8_n128x1` register
structure directly:

- **1 wave / EU** (`llvm.amdgpu-waves-per-eu = "1,1"`): the 256-VGPR
  budget is what makes the reference footprint fit; the default 2-wave cap
  (128 VGPR/wave) spills the FP32 score + O accumulators.
- **Resident Q**: loaded once per work-tile, held in registers across
  every KV block (24 VGPR for FP8 d_qk=192).
- **Single 64-VGPR FP32 score tile** = the QK MFMA accumulator; softmax
  runs in place on it; the FP8 P operand collapses 4:1 IN PLACE into the
  score tile's own low quarter (`v_cvt_pk_fp8_f32 op_sel:[0,0,1]`). No
  separate `p_block`, no score double-buffer.
- **64-VGPR FP32 O accumulator**, eagerly VALU-rescaled per the online
  softmax recurrence.
- **Streamed K band; streamed V band.** K is streamed from LDS into a
  function-local band, consumed by the QK MFMAs, then freed. V is then
  streamed fragment-at-a-time through the band K vacates (disjoint
  lifetimes), the reference lean layout, rather than materialized
  as a whole register tile.

### The reference-exact inner loop

`_attend_exact` lays each KV tile out as 6 barrier-delimited
clusters (7 bare `_s_barrier_raw`) matching the reference's `label_01D6`
boundaries:
C_QK -> C_V_PREFETCH -> C_SOFTMAX_MAX -> C_EXP/rescale -> C_FP8_PACK ->
C_PV, and each cluster comment cites the reference asm line it mirrors.
The two warp-groups (waves 0-3 / waves 4-7) run that body phase-shifted
via an asymmetric +4 prologue `_s_barrier_raw()` stagger, with a
work-split K/V DMA (waves 0-3 produce K, waves 4-7 produce V) into two
disjoint LDS ring regions (K depth-2, V depth-4, the reference V region
is the wider of the two). The shared math is reproduced in-file without
editing any shared file.

The prologue stagger (see the prologue keystone + tail-compensation
comments in `_attend_exact`):
  - `-D exact_stagger` (default = `persistent`): the EXACT
    reference two-half-body discipline: the upper half pays +4 at the
    prologue and the lower half pays a matching +4 at the work-item TAIL
    (the reference `label_06B4` / `label_1A51`), EVERY work-item.
    Per-work-item barrier totals stay EQUAL (no +4N accumulation, no
    deadlock) while the +4 phase skew RE-FORMS each work-item -> a steady
    skew conserved across the CU's whole work stream. Off (the static-grid
    default) = the +4 fires only on wi0; work-items 1..N run in lockstep
    (a no-op at one work-item/CU).
  - `-D v_qktail` (default = NOT `persistent`): prefetch the
    first V band fragments into the QK-tail. A win where registers have
    headroom (static / batch>1); disabled under `persistent` because
    the held-across-softmax band spills at the 256-VGPR ceiling there.

The cadence levers that reproduce the reference instruction schedule (all
unconditional; each pinned by a mask-0 `schedule_barrier`, which fixes a
hand-specified order at codegen; program order alone is re-clustered by
the IGLP solver):
  - Non-materialized V band (lean ~210-VGPR layout): V is streamed
    fragment-at-a-time through the band K vacates (disjoint lifetimes).
  - 4-slot rotating V band in C_PV (3 slots in flight): V reads land ahead
    of their consuming PV MFMA so the per-MFMA drain is soft `lgkmcnt(8)`
    (the reference load/MFMA C_PV cadence, ref asm L744-785).
  - 4-ahead K ring in C_QK pinned by the mask-0 fence (breaks the K-band
    WAR toward the reference soft `lgkmcnt(4)`).
  - Next-tile K/V prefetch issued in C_QK (ref asm L356-410).
  - Resident Q staged DRAM->LDS->VGPR (the reference Q@0x0 region).

### Correctness strategy: reuse the verified MLA-prefill math

The MLA-prefill MATH (QK with nope d=128 + rope d=64; FlashAttention-2
online softmax with running max/sum + cross-tile rescale; in-place FP8
P collapse; PV accumulate; normalize + store; causal / null mask) lives
in `mla_components.mojo`'s `MlaPrefillV2Core[config]` (FP32-scores
path), trimmed to exactly the closure this kernel consumes. Rather than
re-deriving it, `_attend_exact` reuses those BARRIER-FREE numeric
primitives (the `OnlineSoftmax` recurrence, `MhaMmaOp` MFMA/exp/cast
helpers, `_qk_collapse_inplace`, the K/V LDS loaders, `_store_o_to_gmem`)
and `_MlaKDmaPair` for the K DMA, but emits the cluster cadence + the
QK/PV MFMA streams in-file, so the reference's bare `s_barrier` boundaries
are NOT fragmented by the delegated helpers' own `lgkmcnt(0)` drains / IGLP
fences. This file owns (a) the `waves_per_eu=1,1` kernel entry, (b) the
single reference-faithful inner loop, and (c) the host launcher.

Because the `_FP32_SOFTMAX_SCORES` gate (FP8 + KV>=128 + 32x32x64) is the
default-True path for the FP8 KV=128 target shape, every reused
primitive exercises the exact codegen this kernel ships.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.compute.mma import mma as gpu_mma
from max.gpu.host import DeviceContext
from max.gpu.host.compile import CompilationTarget
from max.gpu.sync import s_waitcnt
from std.math import ceildiv
from std.memory import AddressSpace
from std.sys import get_defined_bool, get_defined_int, size_of
from std.sys.intrinsics import readfirstlane
from std.utils import StaticTuple

from layout import TensorEngine, TensorLayout, TileTensor
from layout.coord import Coord, Idx
from layout.tile_layout import row_major
from layout._utils import make_amd_buffer_resource

from structured_kernels.amd_tile_io import (
    RegTile,
    RegTileEpilogue,
    SMemTile,
    _load_from_lds,
    reg_alloc,
    smem_alloc,
)

from nn.attention.mha_mask import CausalMask, MHAMask
from nn.attention.mha_operand import MHAOperand

from .mha_mask import MaskApplier
from .mha_softmax import OnlineSoftmax
from .mha_mma_op import MlaConfigV2, MlaMmaOp
from .mla_components import (
    MlaPrefillV2Core,
    _MlaKDmaPair,
    _sched_barrier_zero,
    _s_barrier_raw,
    _s_setprio,
)


struct MlaPrefillV2[config: MlaConfigV2]:
    """Fresh single-schedule port of the reference integrated MLA-prefill
    inner loop for gfx950. 1 wave / EU; resident Q; single FP32 score
    tile with in-place FP8 P collapse; shared K/V band; 64-VGPR FP32 O
    accumulator with eager rescale; reference work-split K/V DMA + deep
    even wave-spec stagger over a 160 KB LDS.

    AMD FP8 MLA-prefill kernel (FP8 / KV>=128 / 32x32x64). The reused
    numeric closure lives in `mla_components.mojo`
    (`MlaPrefillV2Core`); this file consumes only that module.
    The single reference-exact inner loop (`_attend_exact`) is
    described in the module docstring.

    Parameters:
        config: Shape configuration (`MlaConfigV2`). The FP8 KV=128
            DeepSeek-V3 MLA shape (q_block=32, kv_block=128, depth=128,
            d_qk=192, d_rope=64, cache_depth=576) is the target shape.
    """

    # The verified-math sibling. All numeric steps + the K DMA are
    # delegated to this config-parameterized struct (its static methods
    # are not instance-bound), so `MlaPrefillV2` carries no duplicate
    # math. `MlaPrefillV2Core` stays byte-identical — this file only READS
    # its static methods.
    comptime _Core = MlaPrefillV2Core[Self.config]

    # ---- Config field aliases (mirror MlaPrefillV2Core) -----------------
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth
    comptime D_QK = Self.config.d_qk
    comptime NUM_HEADS = Self.config.num_heads
    comptime NUM_KV_HEADS = Self.config.num_kv_heads
    comptime NUM_WARPS = Self.config.num_warps
    comptime NUM_THREADS = Self.NUM_WARPS * 64
    comptime BM = Self.NUM_WARPS * Self.Q_BLOCK_SIZE

    comptime _MmaOp = MlaMmaOp[Self.config.dtype, Self.config.mha()]

    # Softmax workspace dtype. The reference stays FP32 throughout the
    # softmax (no narrowing) — the FP32 QK MFMA output IS the softmax
    # workspace, telescoped 4:1 to FP8 at the PV boundary. We bind to
    # `MlaPrefillV2Core._SOFTMAX_DTYPE` (NOT a fresh `DType.float32` literal)
    # so the `att_block` / `softmax` types are the SAME comptime SSA
    # expression the delegated `_softmax_tile_fp32` / `_qk_collapse_inplace`
    # helpers expect — the parser cannot fold the deferred gate
    # (`get_defined_bool[...]` chain) to prove a fresh `float32` literal
    # equals it, even though both resolve to FP32 on the target shape.
    comptime _SOFTMAX_DTYPE = Self._Core._SOFTMAX_DTYPE

    # ---- Target-shape predicate -------------------------------------
    # The integrated band-share architecture + the delegated streamed-K
    # / FP32-in-place-softmax math only apply to FP8 + KV>=128 +
    # 32x32x64 (the shape `MlaPrefillV2Core._FP32_SOFTMAX_SCORES` gates on).
    # Enforced in `run` so a mis-shaped config fails at compile time
    # rather than silently routing through math that was never wired
    # for it.
    comptime _IS_DSV_MLA_SHAPE = (
        Self.config.dtype.is_float8()
        and (Self.KV_BLOCK >= 128 or get_defined_bool["allow_kv64", False]())
        and not Self._MmaOp.FP8_MMA_K_128
    )

    # ---- SMEM slot byte geometry (mirror MlaPrefillV2Core) --------------
    comptime _K_ALIGN = (
        Self._MmaOp.K_SUB_ROWS
        * Self._MmaOp.K_SUB_COLS
        * size_of[Self.config.dtype]()
    )
    comptime _V_ALIGN = (
        Self._MmaOp.V_SUB_ROWS
        * Self._MmaOp.V_SUB_COLS
        * size_of[Self.config.dtype]()
    )

    # ===-------------------------------------------------------------=== #
    # R1: reference 3-region LDS (Q@0x0 / K@0xc300 / V@0x18600).
    # ===-------------------------------------------------------------=== #
    # The reference carves the group segment into three DISJOINT regions —
    # Q (resident), K (double-buffered), V (double-buffered)
    # (group_segment_fixed_size 163840). K and V each use a depth-2 double
    # buffer (the producer leads the consumer by one slot — the reference
    # sub-tile offset is < the 1-slot double-buffer depth, so depth-2
    # suffices).
    #
    # The reference K and V regions are NOT the same depth. For the FP8
    # KV=128 shape one K slot = `_K_SLOT_ROWS * K_SUB_COLS = 384*64 =
    # 24576 B` and one V slot = `_V_SLOT_ROWS * V_SUB_COLS = 256*64 =
    # 16384 B`. The reference sizes K @ 49920 B (~= 2 K slots) and V @
    # 64000 B (~= 4 V slots) — the V region is ~4 V slots wide, NOT 2. We
    # therefore DECOUPLE the V ring depth from K's:
    #   - K ring depth-2: 2*24576 = 49152 B (= the reference K region within
    #     its 768-B swizzle pad — the K depth-2 double buffer is correct).
    #   - V ring depth-4: 4*16384 = 65536 B (the nearest integer-slot match
    #     to the reference 64000 B; 64000 is not a multiple of our 16384-B V
    #     slot, so 65536 is as close as slot granularity allows).
    # With the resident-Q LDS region (49152 B) the total is
    # 49152 + 49152 + 65536 = 163840 B = the reference full 160 KB group
    # segment (no slack). Three `smem_alloc`s (K ring, V ring, Q slab) give
    # the three disjoint regions; the ring slots are sub-tiles of each.
    # (Mojo's stack-allocator places the allocations sequentially — it
    # cannot pin the reference exact 0xc300 / 0x18600 byte offsets, but the
    # DISJOINT-region structure + the double buffer ARE what the sub-tile
    # offset needs; the absolute offsets are immaterial to correctness or
    # the ring-aliasing safety.)
    #
    # ===================== BYTE-EXACT LDS: KNOWN TRADEOFF ==============
    # We match the reference LDS STRUCTURE (3 disjoint regions, full-size V)
    # but NOT its byte-exact layout. Two divergences:
    #  1. V SWIZZLE (layout differs; bank-conflict EQUIVALENT):
    #     The reference applies an ADDITIVE-skew LDS swizzle to V (per-key
    #     0x410/0x820 stride, 256-B read pitch; ref asm L46-93);
    #     ours uses `v_swizzle = None` (identity; the V producer is the
    #     `SubTileLoaderLDS_st_8x32` DMA, no swizzle param).
    #     The LAYOUTS differ, but they are bank-conflict EQUIVALENT: V is
    #     read via `ds_read_b64_tr_b8` (a TRANSPOSE read), whose hardware
    #     lane->bank grouping is set by the transpose pattern, NOT the
    #     contiguous `(addr>>2)&31` cycle — so a byte-address swizzle only
    #     relabels within the same conflict class. The rocprofv3
    #     `LdsBankConflict` counter is identical with swizzle OFF vs
    #     `Swizzle(1,4,6)` / `Swizzle(1,5,5)`; applying one breaks the
    #     reference v227 single-base CSE for 0 benefit. NOT a divergence to
    #     fix. (`Swizzle(1,4,10)` is a no-op here: bit14 = the 16KB slot
    #     size, above the V-slot address range.)
    #  2. V granularity: the reference V region is 64000 B; our V ring is an
    #     integer count of 16384-B slots (depth-4 = 65536 B, nearest match).
    # ==================================================================
    comptime _RING_DEPTH = 2

    # ---- Reference V adapter (write + read) -- `-D v_full_v227` ----
    # The reference-faithful V LDS layout, the default-on adapter for this
    # research kernel (production `MlaPrefillV2Core` passes `v_full_v227=False`
    # to the shared fns and is byte-identical to its non-adapter codegen).
    # It drives BOTH halves of the adapter `W∘R`:
    #   W = the producer `SubTileLoaderLDS_st_8x32[v_full_v227=True]` (via
    #       `_dma_v`) reorganizes the cooperative `buffer_load…lds` into the
    #       reference 16-chunk LDS layout (chunk stride 0x410 = the reference M0
    #       step), reading OUR ragged key-major V — SAME 16-burst count +
    #       1024-B granularity as the default `st_8x32` fill (no DMA inflation).
    #   R = `precompute_v_lane_base[v_full_v227=True]` (the reference `v227`
    #       per-lane base) + `load_V_frag[v_full_v227=True]` (the faithful
    #       readout cell `i_strip*0x2080 + j_depth*0x20 + r*0x100`).
    # `W` is `pi o W_ours`, the LDS-byte permutation `pi: ours_read_addr ->
    # ref_read_addr` PROVEN a bijection over all 16384 slot bytes (the tr8
    # transpose cancels — both reads issue the identical `ds_read_tr8_b64`, so
    # only the LDS address differs). The slot grows 16384 → 16640 B
    # (`_V_SLOT_PAD_ROWS=4`; the v227 read reaches byte 16623) and the V ring
    # drops to depth-3 (the +256 B/slot makes depth-4 overflow the 160 KB
    # group segment). The adapter is CORRECTNESS-CLEAN: cos_sim with the
    # adapter on is bit-identical to off, and it makes the V transpose read
    # bank-conflict-free.
    #
    # PERF is a batch crossover: the bank-conflict win lands net-positive once
    # enough waves co-reside to hide the chunk-stepped producer's per-chunk M0
    # setup (the larger-batch grids), but the under-occupied b1 grid exposes
    # that setup latency (it is NOT a spill — the adapter does not raise the
    # VGPR count, 0 spill both). Default ON is intentional; toggle off with
    # `-D v_full_v227=false`.
    comptime _V_FULL_V227 = get_defined_bool["v_full_v227", True]()
    # Spell the `v_full_v227` V LDS adapter (both WRITE producer and READ
    # base) via CuTe Layout Algebra (`crd2idx` over per-bit `Coord`s) instead
    # of hand-rolled runtime bit arithmetic. SAME geometry, different spelling
    # — both the WRITE source byte and the READ per-lane base are bit-LINEAR
    # over the bit-decomposed (chunk, lane) index, so the bit-permutation +
    # skews become a declarative Coord stride table; cos_sim is bit-identical
    # to the hand path (the gate is correctness-equivalence, NOT a codegen
    # win — `crd2idx`'s generic divmod machinery is heavier than the minimal
    # hand bit-ops, so it is GATED OFF by default). Default OFF (hand path).
    # `-D v227_layout=true` selects the Layout spelling on BOTH the WRITE
    # path (read directly in `SubTileLoaderLDS_st_8x32`) and the READ path
    # (threaded into `precompute_v_lane_base` below) — a clarity/enablement
    # choice, not a codegen win. Only consulted when `_V_FULL_V227` is True.
    comptime _v227_layout = get_defined_bool["v227_layout", False]()
    # The adapter writes the reference chunk-stepped layout, whose 16 chunks
    # at the 0x410 (1040-B) stride span [0, 16624) — 240 B past the natural
    # 16384-B slot. The reference padded V slot is 0x4100 = 16640 B =
    # 16 * 0x410 = `_V_SLOT_ROWS(256) + 4` FP8 rows (256 B pad), so allocate
    # 4 extra rows. Off → 0 pad rows (byte-identical natural slot).
    comptime _V_SLOT_PAD_ROWS = 4 if Self._V_FULL_V227 else 0
    # V ring depth: the +256 B/slot pad makes depth-4 padded (66560 B)
    # overflow the 160 KB group segment (K 49152 + V 66560 + Q 49152 =
    # 164864 > 163840). Drop to depth-3 when running the adapter (49920 B →
    # total 148224, fits) — still ≥ the depth-2 the V double buffer requires.
    # Off keeps depth-4 ("nearest match to the reference 64000-B V region").
    comptime _V_RING_DEPTH = 3 if Self._V_FULL_V227 else 4

    comptime _K_SLOT_ROWS = Self._Core._K_SLOT_ROWS
    # Padded per-slot row count (= _V_SLOT_ROWS + per-block pad rows). For
    # the unpadded path this is exactly `_Core._V_SLOT_ROWS` (byte-identical).
    comptime _V_SLOT_ROWS = Self._Core._V_SLOT_ROWS + Self._V_SLOT_PAD_ROWS
    comptime _K_SUB_COLS = Self._MmaOp.K_SUB_COLS
    comptime _V_SUB_COLS = Self._MmaOp.V_SUB_COLS

    # Whole-ring SMEM layouts: `_RING_DEPTH` K slots stacked rows-major
    # in ONE allocation (the disjoint K region), likewise for V. A ring
    # slot is `k_ring.tile[_K_SLOT_ROWS, _K_SUB_COLS](slot, 0)` — a pure
    # ptr offset, byte-identical in shape to the per-slot `smem_layout_k`
    # the shared `_MlaKDmaPair` / V loaders consume.
    comptime _K_RING_LAYOUT = row_major[
        Self._RING_DEPTH * Self._K_SLOT_ROWS, Self._K_SUB_COLS
    ]()
    comptime _V_RING_LAYOUT = row_major[
        Self._V_RING_DEPTH * Self._V_SLOT_ROWS, Self._V_SUB_COLS
    ]()

    # ---- Reference Q-in-LDS resident region (Q@0x0). ----------------
    # The reference stages Q in its own LDS region (Q@0x0, 49920 B) — DMA'd
    # DRAM->LDS once per work-tile (`buffer_load_dwordx4 ... lds`), then
    # `ds_read` into v[4:27] ONCE, resident across all KV iters. Reproduced
    # here as ONE workgroup-wide FLAT `[BM, D_QK]` FP8 slab (no swizzle — Q
    # is read flat by `_Core.load_q`'s per-lane addressing), each warp owning
    # rows `[w_id*Q_BLOCK_SIZE, +Q_BLOCK]`. For FP8 KV=128: BM=256 * D_QK=192
    # = 49152 B (~= the reference 49920 B Q region). The flat slab makes the
    # `ds_read` mirror `load_q`'s DRAM addressing exactly (only the row
    # stride differs: D_QK contiguous in LDS vs `_Q_ROW_STRIDE` in DRAM),
    # so the consumed `q_reg` fragment is byte-identical to a DRAM load.
    comptime _Q_LDS_LAYOUT = row_major[Self.BM, Self.D_QK]()

    # Per-(batch, head) 2D view layout types (mirror MlaPrefillV2Core).
    comptime _QPerHeadLayoutT = Self._Core._QPerHeadLayoutT
    comptime _OPerHeadLayoutT = Self._Core._OPerHeadLayoutT
    comptime _Q_LAYOUT_MLA_T = Self._Core._Q_LAYOUT_MLA_T
    comptime _ATT_LAYOUT_T = Self._Core._ATT_LAYOUT_T
    comptime _O_LAYOUT_T = Self._Core._O_LAYOUT_T
    comptime _O_T_LAYOUT_T = Self._Core._O_T_LAYOUT_T

    # ---- PV strip geometry (mirror MlaPrefillV2Core) --------------------
    # For FP8 KV=128: `_NUM_PV_SUBTILES = KV_BLOCK / MMA_K = 128/64 = 2`
    # MMA_K-row PV strips per V tile; each PV fragment is `_PV_A_FRAG`
    # FP8 elements (one V band slot, transpose-read via `load_V_frag`).
    comptime _PV_A_FRAG = Self._Core._PV_A_FRAG
    comptime _NUM_PV_SUBTILES = Self._Core._NUM_PV_SUBTILES

    # ===-------------------------------------------------------------=== #
    # Reference-exact inner loop.
    # ===-------------------------------------------------------------=== #
    # The reference-faithful mainloop: a 6-cluster barrier-delimited inner
    # loop that emits the reference cluster cadence directly, rather than
    # delegating the softmax / PV body to `Self._Core._*`.
    #
    # WHY in-file (not via the body helpers): the helpers
    # (`_softmax_tile_fp32` / `_pv_whole` / `_qk_with_kreg_mla_nope`) carry
    # their own `lgkmcnt(0)` full-drains + IGLP fences, which fragment the
    # MFMA stream. The reference pays its per-sub-iter `s_barrier`s BARE (no
    # `lgkmcnt(0)`, no `sched_barrier(0)`, no adjacent branch) so the MFMA
    # stream flows; a faithful replica emits those bare boundaries itself
    # while reusing only the BARRIER-FREE numeric primitives.
    #
    # The cluster math is correct at the FP8 KV=128 target: real
    # `v_mfma_f32_32x32x64_f8f6f4` QK/PV, online-softmax max/exp/rescale,
    # in-place FP8 P collapse, streamed V. NOT yet reference-structural: the
    # dedicated masked-tail cluster + the 4-barrier resync epilogue (we use
    # the causal `max_num_tiles` bound + the verified
    # `normalize_output`/`_store_o_to_gmem` instead -- correct, not yet
    # reference-shaped). See the MASKED TAIL + EPILOGUE note at the loop end.
    #
    # The 6 clusters per sub-iter delimited by bare `s_barrier`s map to the
    # reference waves-0-3 asm:
    #   C_QK_MFMA       -> L411 s_barrier
    #   C_V_PREFETCH    -> L429 s_barrier + setprio0 + L431 s_barrier
    #   C_SOFTMAX_MAX   -> L504 s_barrier
    #   C_EXP_RESCALE   -> L707 s_barrier
    #   C_FP8_PACK      -> L740 s_barrier + (vmcnt0 lgkmcnt8) + setprio1
    #                                     + L744 s_barrier
    #   C_PV_MFMA       -> back-edge
    # plus the prologue stagger on the upper wave-half (the conserved-
    # offset ping-pong keystone).
    @staticmethod
    @always_inline
    def _load_q_lds_exact[
        layout: TensorLayout,
    ](
        q_warp_2d: TileTensor[Self.config.dtype, layout, ...],
        q_lds: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        w_id: Int,
        scale_log2e: Float32,
    ) -> RegTile[Self.config.dtype, Self._Q_LAYOUT_MLA_T, MutUntrackedOrigin]:
        """Reference Q-in-LDS resident load (Q@0x0 region pattern).

        Stages Q DRAM->LDS->VGPR, producing the SAME `q_reg` fragment a
        direct DRAM->VGPR load (`_Core._load_q_and_scale_mla`) produces, so
        the QK-MFMA consumer is unchanged.

        Steps (FP8 e4m3 only, the target shape):
        1. Cooperative DMA: each warp DMAs its `q_warp_2d` row-block
           (Q_BLOCK_SIZE x D_QK) into its LDS slot
           `q_lds.tile[Q_BLOCK_SIZE, D_QK](w_id, 0)` via
           `buffer_load_*_lds`. Each lane issues the `D_QK`-wide row it
           will later read (`row = lid % Q_BLOCK_SIZE`), so the DMA +
           ds_read share the same per-lane row.
        2. Hard drain (`s_waitcnt vmcnt(0)`) + workgroup `s_barrier`:
           Q is loaded ONCE in the prologue, so the hard drain is free
           (not in the KV hot loop), and it is the alias-scope-safe
           fence required for the un-scoped DMA -> ds_read handshake: a
           runtime `vmcnt(0)+s_barrier` is mandatory when the loader is
           not `async_copies`-tagged, which `load_to_lds` is not here.
        3. `ds_read` Q from the FLAT LDS slot into `q_reg`, mirroring
           `_Core.load_q`'s FP8 per-lane addressing (`row = lid % 32`,
           `col_base = (lid // 32) * 32`, two 16-B halves per K-tile)
           but with the contiguous LDS row stride `D_QK`.
        4. Prescale (comptime-elided for FP8, post-QK scale, matching
           `_Core._load_q_and_scale_mla`).

        Constraints:
            FP8 e4m3 only (the KV=128 target). BF16 falls back to
            the DRAM path at the call site.
        """
        comptime assert (
            Self.config.dtype == .float8_e4m3fn
        ), "MlaPrefillV2._load_q_lds_exact: FP8 e4m3 only"

        comptime _BK = Self._MmaOp.MMA_K
        comptime _num_k_tiles = Self.D_QK // _BK
        comptime _FE = Self._Q_LAYOUT_MLA_T.static_shape[2]
        comptime _HALF = _FE // 2  # 16 FP8 = 16 B per ds_read

        var lid = Int(lane_id())

        # The warp's flat LDS slot: rows `[w_id*Q_BLOCK_SIZE, +Q_BLOCK]`
        # of the workgroup `[BM, D_QK]` slab.
        var q_slot = q_lds.tile[Self.Q_BLOCK_SIZE, Self.D_QK](w_id, 0)

        # ---- (1) Cooperative DMA Q DRAM -> LDS ------------------------
        # Canonical `buffer_load_*_lds` lane->LDS mapping (mirrors
        # `SubTileLoaderLDS.load_tile`, amd_tile_io.mojo:1483):
        #   - `shared_ptr` is WAVE-UNIFORM (`readfirstlane`) — ONE LDS
        #     base, advanced per row.
        #   - `vector_offset` is the PER-LANE DRAM source offset; the
        #     hardware writes lane `l`'s `width` elts to LDS at
        #     `shared_ptr + (vector_offset - uniform_base)`.
        #   - `scalar_offset` is the wave-uniform DRAM anchor.
        # CRITICAL: the DRAM Q warp tile is STRIDED — row stride is
        # `_Q_ROW_STRIDE = NUM_HEADS * D_QK` (Q is `[seq, NUM_HEADS,
        # D_QK]`), NOT contiguous. A flat element walk over the buffer
        # resource therefore crosses the inter-row gap (3072 vs 192) and
        # reads garbage past row 0 (first attempts: cos 0.57 — row 0
        # correct, rest wrong). Issue PER ROW: within one row both DRAM
        # (D_QK contiguous) and LDS (D_QK contiguous) match, so the LDS
        # slot becomes a faithful row-major `[Q_BLOCK_SIZE, D_QK]` copy
        # = exactly what the step-(3) flat ds_read assumes.
        comptime _q_src_row_stride = type_of(q_warp_2d).static_stride[0]
        comptime _DQK_HALVES = Self.D_QK // _HALF  # cols per row / lane width
        comptime assert (
            _DQK_HALVES <= WARP_SIZE
        ), "MlaPrefillV2._load_q_lds_exact: D_QK row exceeds warp width"
        var q_bc = make_amd_buffer_resource(q_warp_2d)
        # Lane `lid` (lid < _DQK_HALVES) owns col-chunk `lid` of every
        # row; lanes >= _DQK_HALVES are gated out (no over-read).
        comptime for r in range(Self.Q_BLOCK_SIZE):
            var q_row_smem = readfirstlane(
                rebind[
                    UnsafePointer[
                        Scalar[Self.config.dtype],
                        MutAnyOrigin,
                        address_space=.SHARED,
                    ]
                ](q_slot.tile[1, Self.D_QK](r, 0).ptr)
            )
            if lid < _DQK_HALVES:
                # Split DRAM source into wave-uniform anchor
                # (`scalar_offset` = the strided DRAM row) + per-lane
                # `vector_offset` (the in-row col chunk). The per-lane
                # LDS write lands at `q_row_smem + vector_offset` (in-row,
                # contiguous); the DRAM read is `scalar_offset +
                # vector_offset` (strided row + col).
                q_bc.load_to_lds[Self.config.dtype, width=_HALF](
                    Int32(lid * _HALF),
                    q_row_smem,
                    scalar_offset=Int32(r * _q_src_row_stride),
                )

        # ---- (2) Hard drain + barrier (free: prologue, once) ---------
        s_waitcnt[vmcnt=UInt32(0)]()
        _sched_barrier_zero()
        _s_barrier_raw()
        _sched_barrier_zero()

        # ---- (3) ds_read Q from LDS into q_reg -----------------------
        # Mirror `_Core.load_q` FP8 addressing with the LDS row stride.
        var q_reg = reg_alloc[Self.config.dtype](Self._Core.Q_LAYOUT_MLA)
        var q_reg_v = q_reg.vectorize[1, 1, _HALF]()
        # Per-lane Q-row base as a tile COORDINATE, not raw `.ptr +`
        # arithmetic: lane `lid` owns MFMA row `lid % 32` of the flat
        # `[Q_BLOCK_SIZE, D_QK]` LDS slot, starting at col-block
        # `lid // 32` (the half-warp owns the upper or lower 32-col
        # block — the FP8 32x32x64 B-operand lane layout). The
        # `.tile[1, 32](row, col_block)` view's `.ptr` is exactly
        # `q_slot.ptr + (lid % 32)*D_QK + (lid // 32)*32`, replacing the
        # hand `row_offset * D_QK + col_base` offset.
        var q_lane = q_slot.tile[1, 32](lid % 32, lid // 32)
        comptime _ESZ = size_of[Self.config.dtype]()
        # The per-K-tile / per-half stride (`j*_BK`, `+_HALF`) is COMPTIME,
        # so it folds into `ds_read offset:imm` via `_load_from_lds`'s
        # `typed_imm_offset_bytes` — the "one hoisted base + per-cell
        # comptime immediate" idiom (amd_tile_io.mojo:663, codegen-neutral
        # vs the per-cell `.ptr + offset` form).
        comptime for j in range(_num_k_tiles):
            var lo = _load_from_lds[
                width=_HALF, typed_imm_offset_bytes=(j * _BK) * _ESZ
            ](q_lane.ptr)
            var hi = _load_from_lds[
                width=_HALF, typed_imm_offset_bytes=(j * _BK + _HALF) * _ESZ
            ](q_lane.ptr)
            q_reg_v[0, j, 0] = rebind[type_of(q_reg_v[0, j, 0])](lo)
            q_reg_v[0, j, 1] = rebind[type_of(q_reg_v[0, j, 1])](hi)

        # ---- (4) Prescale (FP8: comptime-elided, post-QK scale) ------
        comptime if Self._Core.prescale_q:
            comptime _H = Self._Q_LAYOUT_MLA_T.static_shape[0]
            comptime _W = Self._Q_LAYOUT_MLA_T.static_shape[1]
            var q_v = q_reg.vectorize[1, 1, _FE]()
            comptime for h in range(_H):
                comptime for w in range(_W):
                    q_v[h, w, 0] = (
                        q_v[h, w, 0].cast[.float32]() * scale_log2e
                    ).cast[Self.config.dtype]()

        return q_reg

    @staticmethod
    @always_inline
    def _attend_exact[
        k_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        out_dtype: DType,
        o_layout: TensorLayout,
        o_engine: TensorEngine,
        //,
    ](
        mut q_reg: RegTile[
            Self.config.dtype, Self._Q_LAYOUT_MLA_T, MutUntrackedOrigin
        ],
        mut o_reg: RegTile[.float32, Self._O_LAYOUT_T, MutUntrackedOrigin],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
        mask_functor: mask_t,
        mut att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        k_ring: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        v_ring: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        k_op: k_t,
        v_op: v_t,
        o_warp_2d: TileTensor[
            out_dtype, o_layout, MutAnyOrigin, Engine=o_engine
        ],
        num_tiles: Int,
        max_num_tiles_local: Int,
        tile_idx: Int,
        start_pos: Int,
        head_idx: Int,
        batch_idx_u32: UInt32,
        kv_head_idx_u32: UInt32,
        w_id: Int,
        l_id: Int,
        block_tile_idx: Int,
        seq_len: Int,
        scale_log2e: Float32,
        stagger: Bool,
        apply_prologue_stagger: Bool,
    ):
        """Reference-exact 6-cluster barrier-delimited mainloop.

        The body is the reference waves-0-3 cluster structure with the
        upper-half +4 prologue stagger; the cluster math is correct at the
        FP8 KV=128 target and emits the bare-barrier cadence directly
        rather than delegating to `Self._Core._softmax_tile_fp32` /
        `_pv_whole` / `_qk_with_kreg_mla_nope` (which carry their own drain
        walls). The cadence levers (non-materialized V, 4-slot V band,
        4-ahead K ring, C_QK prefetch, all pinned by the mask-0 fence)
        reproduce the reference instruction schedule. The masked tail +
        4-barrier resync epilogue are the remaining reference-structural
        follow-ons (see the loop-end note).
        """

        # ---- Work-split K/V DMA helpers (REUSED low-level primitives) -
        # The double-issue warp-remap cooperative loaders
        # (`_MlaKDmaPair.dma` / `_Core._dma_v`) fill the disjoint K/V LDS
        # double-buffer regions in the prologue + per-tile prefetch. The
        # reference work-split: waves 0-3 produce K, waves 4-7 V. (DMA is a
        # reused primitive, not body delegation — it does not impose
        # cluster-boundary drains.)
        var w_remap = w_id & 3

        @__parameter
        @always_inline
        def _dma_k_into(slot: Int, t: Int):
            var kp = _MlaKDmaPair[Self.config](
                k_op, batch_idx_u32, kv_head_idx_u32, t
            )
            var k_slot = k_ring.tile[Self._K_SLOT_ROWS, Self._K_SUB_COLS](
                slot, 0
            )
            kp.dma(k_slot, w_remap, l_id)
            kp.dma(k_slot, w_remap + 4, l_id)

        @__parameter
        @always_inline
        def _dma_v_into(slot: Int, t: Int):
            var v_slot = v_ring.tile[Self._V_SLOT_ROWS, Self._V_SUB_COLS](
                slot, 0
            )
            Self._Core._dma_v[v_full_v227=Self._V_FULL_V227](
                v_slot, v_op, batch_idx_u32, kv_head_idx_u32, t, w_remap, l_id
            )
            Self._Core._dma_v[v_full_v227=Self._V_FULL_V227](
                v_slot,
                v_op,
                batch_idx_u32,
                kv_head_idx_u32,
                t,
                w_remap + 4,
                l_id,
            )

        @__parameter
        @always_inline
        def _dma_kv_split(k_slot_idx: Int, v_slot_idx: Int, t: Int):
            # Reference work-split: phase-lagged upper half (waves 4-7)
            # produces V, lower half (waves 0-3) produces K. K and V use
            # DECOUPLED ring depths (K depth-2, V depth-4), so the producing
            # half indexes its own ring's slot.
            if stagger:
                _dma_v_into(v_slot_idx, t)
            else:
                _dma_k_into(k_slot_idx, t)

        # Reference producer-leads-consumer-by-one-slot prefetch distance:
        # the K/V double buffer publishes slot `(t+1)%RING` while the
        # consumer reads slot `t%RING`. K uses `slot = t%_RING_DEPTH`, V uses
        # its own `v_slot_idx = t%_V_RING_DEPTH` (decoupled ring depths). Both
        # rings must exceed the prefetch distance for the producer/consumer
        # slots to be distinct.
        comptime _PF_DIST = 1
        comptime assert (
            Self._RING_DEPTH > _PF_DIST
        ), "MlaPrefillV2._attend_exact: _RING_DEPTH must exceed pf dist"
        comptime assert (
            Self._V_RING_DEPTH > _PF_DIST
        ), "MlaPrefillV2._attend_exact: _V_RING_DEPTH must exceed pf dist"

        # =================================================================
        # PROLOGUE — work-split DMA of the first `_PF_DIST` KV tiles into
        # ring slots `0.._PF_DIST-1`, then the 4-barrier upper-half stagger.
        # =================================================================
        comptime for _pf in range(_PF_DIST):
            if _pf < num_tiles:
                # Prologue primes slot `_pf` in each ring. For `_PF_DIST=1`
                # both K and V prime slot 0; the modulo keeps it correct if
                # `_PF_DIST` ever exceeds a ring depth.
                _dma_kv_split(
                    _pf % Self._RING_DEPTH, _pf % Self._V_RING_DEPTH, _pf
                )

        # ---- THE PROLOGUE STAGGER KEYSTONE ----------------------------
        # Reproduces the reference prologue barrier offset EXACTLY. The
        # reference has NO pre-fork barrier; it forks at asm L261
        # (`s_cmp_lt_i32 s62, 4`) into group A (waves 0-3) and group B
        # (waves 4-7), and each group's prologue hits a fixed number of bare
        # `s_barrier`s before its first steady K read (`ds_read v226`):
        #   - group A (waves 0-3): asm L336 + L344            = 2 barriers
        #   - group B (waves 4-7): asm L1296,L1303,L1305-1308 = 6 barriers
        #   => lower half 2, upper half 6, OFFSET = +4.
        # The offset is conserved by hardware through every shared body
        # barrier (CDNA4 `s_barrier` is a counting barrier), establishing the
        # constant phase skew between the two 4-warp groups.
        #
        # We emit the SAME absolute counts. Our shared prologue already has
        # TWO barriers both halves hit: the Q-load drain (the `_s_barrier_raw`
        # after the Q DMA) and the cold-start KV-DMA sync below. So the lower
        # half is ALREADY at 2 (== reference group A). The upper half then
        # takes `_STAGGER_EXTRA = 4` more -> 6 total (== reference group B),
        # for the +4 offset. Each extra is BARE (`_s_barrier_raw`) +
        # reorder-fenced, matching the reference bare prologue barriers (a
        # pure phase-shift, no drain the sibling can't hide). The DMA above is
        # drained ONCE before the shared barrier so neither half reads
        # K[0]/V[0] before it lands.
        #
        # NOTE on the OFFSET as a perf lever: this stagger is perf-INERT on
        # the current body because the body is LOCKSTEP (no `s_cmp...4`
        # two-stream fork; the `comptime if is_upper` DMA splits do not
        # diverge the body in codegen, so both halves execute the same
        # instruction at each shared barrier and the offset cannot create
        # A-QK-in-B-PV overlap). The offset is matched here for reference
        # barrier FIDELITY, not as a throughput change; the productive
        # two-stream overlap requires the warp-group fork, which is a
        # separate, larger structural change.
        comptime _STAGGER_EXTRA = 4
        # Shared cold-start sync (both halves) — drains the prologue DMA.
        # vmcnt-only: the prologue Q/K/V loads are buffer_load->LDS (VMEM-
        # counted), so vmcnt(0) alone guarantees they landed before the barrier.
        # The reference cold-start is vmcnt-only too — no ds_read has been
        # issued yet, so there is no LDS traffic to drain; an lgkmcnt(0) here
        # would be a spurious hard drain paid once per workgroup.
        s_waitcnt[vmcnt=UInt32(0)]()
        _sched_barrier_zero()
        _s_barrier_raw()
        _sched_barrier_zero()
        # Upper-half-only +4 (the asymmetric stagger). Bare barriers; no
        # drain (the shared sync above already retired the DMA), so this is
        # a pure phase-offset injection.
        #
        # TWO firing disciplines, selected by `-D exact_stagger`:
        #
        # (A) Default (`exact_stagger=False`): `apply_prologue_stagger`-
        #     gated -> the upper +4 fires ONCE per CU (the first work-item).
        #     Rationale: re-injecting +4 per work-item with NO compensating
        #     lower-half barrier would ACCUMULATE (+4N) and drift the
        #     ping-pong. So this confines the offset to wi0 and runs wi1..N in
        #     lockstep. The unmatched +4 on wi0 survives via the ENDPGM
        #     exception (no tail compensation). The static grid (one work-item
        #     per block) always applies it.
        #
        # (B) `exact_stagger=True`: the EXACT reference two-half-body
        #     discipline. The reference persistent loop re-enters `label_011C`
        #     (lower, 2 prologue barriers) / `label_06B4` (upper, 6 = 2 + 4)
        #     EVERY work-item, and the lower half pays a matching +4 at
        #     `label_1A51` (the work-item TAIL) EVERY work-item before the
        #     `s_branch label_011C` back-edge. So per work-item the totals are
        #     EQUAL (upper = 6+7T prologue-heavy, lower = 6+7T tail-heavy) and
        #     the +4 skew RE-FORMS each work-item (upper races ahead at the new
        #     prologue, lower catches up at the new tail). NO accumulation: the
        #     per-work-item balance prevents +4N drift. This is NOT the wi0-
        #     confined construction — the offset is conserved as a steady
        #     phase skew across the CU's whole work stream, which is the
        #     overlap-producing structure. The matching lower-half tail +4 is
        #     emitted after `_store_o_to_gmem` below (also `_EXACT_STAGGER`-
        #     gated, every work-item).
        # `exact_stagger` / `v_qktail` default off `persistent`:
        # exact-stagger ON only in the persistent grid (the conserved phase skew
        # lives there); v_qktail ON everywhere EXCEPT persistent (it spills at
        # the 256-VGPR ceiling there). Both stay overridable via `-D`.
        comptime _PERSISTENT = get_defined_bool["persistent", False]()
        comptime _EXACT_STAGGER = get_defined_bool[
            "exact_stagger", _PERSISTENT
        ]()
        comptime if _EXACT_STAGGER:
            # EVERY work-item, upper half only (drops the `apply_prologue
            # _stagger` gate -> the reference per-work-item re-formation).
            if stagger:
                comptime for _s in range(_STAGGER_EXTRA):
                    _sched_barrier_zero()
                    _s_barrier_raw()
                    _sched_barrier_zero()
        else:
            # Default: wi0-only.
            if stagger and apply_prologue_stagger:
                comptime for _s in range(_STAGGER_EXTRA):
                    _sched_barrier_zero()
                    _s_barrier_raw()
                    _sched_barrier_zero()

        var mask = MaskApplier[
            Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE, KV_BLOCK_SIZE=Self.KV_BLOCK
        ](mask_functor)

        # 32-bit loop induction (scalar guard, zero VGPRs — the spill-
        # avoidance lever the verified `MlaPrefillV2Core` path relies on).
        var max_num_tiles_i32 = Int32(max_num_tiles_local)
        var num_tiles_i32 = Int32(num_tiles)
        var ring_i32 = Int32(Self._RING_DEPTH)
        var v_ring_i32 = Int32(Self._V_RING_DEPTH)

        # ============================================================== #
        # CADENCE — cross-cluster K read-pipelining.
        # ============================================================== #
        # The reference keeps its last PV MFMAs SOFT (`lgkmcnt(6/4)`, asm
        # L776-790) and its next-tile QK seed MFMA SOFT (`lgkmcnt(4)`, asm
        # L786) by issuing the NEXT tile's first K-band `ds_read`s (LDS->reg)
        # INTO the current PV-tail (`ds_read_b128 v226 offset:24960` = the
        # OTHER K double-buffer slot). Our collapsed body instead closes C_PV
        # with the LDS-read pipe empty (next-tile prefetch is on the vmcnt
        # `buffer_load...lds` path, drained before PV), so the last PV MFMA +
        # the next QK seed both hit hard `lgkmcnt(0)`.
        #
        # FIX: RELOCATE the next tile's first `_PF_NEXT_K` K ring-fragment
        # `load_K_frag` (LDS->reg) `ds_read`s into THIS tile's PV-tail/QK-tail,
        # pinned by the mask-0 `_sched_barrier_zero()` reorder fence. The
        # relocated reads:
        #   (a) keep `lgkmcnt` nonzero across the last PV MFMAs  -> PV-tail
        #       softens (reads outstanding when the MFMA retires), AND
        #   (b) are HANDED to the next tile's QK seed via the closure-scope
        #       `_nxt_kf*` vars (the cross-call hand-off is FREE here:
        #       `_one_tile_exact` is a closure over THIS scope, so a `var`
        #       declared here persists across BOTH per-back-edge calls) ->
        #       QK-head softens (the seed consumes a read issued a full PV
        #       cluster earlier).
        # The read MUST move in program order (a fence WITHOUT relocation is
        # inert); the PV source is edited, not QK, because the fence pins
        # INTRA-region but cannot drag a read ACROSS the PV->QK region split
        # (the V-in-PV finding).
        # Next-K DMA vs the relocated ds_read race: the next-tile K DMA into
        # `oslot` is issued in C_QK and drained by the FP8 cluster's
        # `s_waitcnt vmcnt(0)` BEFORE PV — so the PV-tail `ds_read` of `oslot`
        # reads landed next-tile K. Bit-exact.
        # The reference-faithful cross-cluster held-register schedule, now
        # UNCONDITIONAL (no `-D` gating): hold the next tile's first 2 K
        # fragments (nope cols 0,1) across the PV->QK seam and issue them as
        # UNCONDITIONAL-SSA reads in the PV-tail (col 0) and QK-tail (col 1),
        # so neither tail MFMA's K operand is the freshest LDS read -> the
        # `lgkmcnt(0)` drains soften to `lgkmcnt(4)`. The conditional (runtime
        # `if _nxt_k_valid` / `if _do_pf`) variant does NOT work — it PHI-
        # rematerializes the K read at the control-flow merge and keeps the
        # hard drain. The hand-off is PRIMED from K slot 0 before the loop so
        # the QK-prologue consume is unconditional (no fresh-load arm for RA
        # to fall back to).
        #
        # `_PF_NEXT_K` = 2 held cross-seam K cols (slots 0,1 are nope frags,
        # read off the slot view directly — no rope sub-view). The relocated
        # next-tile K `ds_read`s read `oslot` (the next K double-buffer slot),
        # already DMA'd in C_QK and drained `vmcnt(0)` at the FP8 cluster
        # before PV. Reads on the last tile hit a stale slot but the value
        # reaches the next QK seed only via `_nxt_kf*`, consumed only when a
        # next tile exists -> bit-exact, no PHI, no merge drain.
        comptime _PF_NEXT_K = 2
        comptime _PF_FE = Self._MmaOp.FRAG_ELTS
        comptime _PF_FRAG = SIMD[Self.config.dtype, _PF_FE]
        # PV-tail soften: 1 trailing PV MFMA whose refill region carries
        # the next-tile K read (col 0).
        comptime _PF_PV_TAIL = 1
        # QK-tail soften: 1 trailing QK MFMA whose refill region carries
        # the next-tile K read (col 1, a genuinely-live seed value so DCE
        # cannot remove the dead-store read — the PV-tail owns col 0).
        comptime _PF_QK_TAIL = 1
        comptime assert (
            _PF_QK_TAIL <= Self._Core._NUM_K_NOPE_TILES
        ), "PV-tail/QK-tail prefetch reads nope frags only"
        # The relocated next-tile K fragments (cols 0,1 of the next call's QK
        # ring), written in tile t's PV-tail (col 0) + QK-tail (col 1),
        # consumed by tile t+1's QK prologue. 2 frags x 8 VGPR = 16 VGPR held
        # across the back-edge.
        var _nxt_kf0 = _PF_FRAG(0)
        var _nxt_kf1 = _PF_FRAG(0)

        # =================================================================
        # SUB-ITER BODY — the 6 clusters / 7 bare `s_barrier`s.
        # =================================================================
        # One call = ONE KV tile = ONE reference sub-iter. The back-edge
        # driver below calls it TWICE per back-edge (the reference 2-sub-iter
        # unroll). The 7 `_s_barrier_raw()`s are at the reference cluster
        # boundaries (asm L411/429/431/504/707/740/744). Each is BARE +
        # reorder-fenced so the cadence survives codegen at the boundary, and
        # the cluster interior stays free of `lgkmcnt(0)` / `sched_barrier(0)`
        # walls.
        @__parameter
        @always_inline
        def _one_tile_exact[is_upper: Bool](t32_arg: Int32):
            # Reference-faithful non-materialized V (the lean band layout):
            # V is streamed fragment-at-a-time through the (post-QK dead) K
            # band instead of materializing the whole 64-VGPR `V_LAYOUT`
            # tile held live across softmax. Wired in C_V_PREFETCH (no
            # whole-tile V load) and C_PV_MFMA (JIT `load_V_frag` per strip).
            var t = Int(t32_arg)
            var slot = Int(readfirstlane(t32_arg % ring_i32))
            var oslot = Int(
                readfirstlane((t32_arg + Int32(_PF_DIST)) % ring_i32)
            )
            # V uses a DEEPER ring (depth-4) than K (depth-2), so its
            # consume / prefetch slots are computed against `_V_RING_DEPTH`.
            var v_slot_idx = Int(readfirstlane(t32_arg % v_ring_i32))
            var v_oslot = Int(
                readfirstlane((t32_arg + Int32(_PF_DIST)) % v_ring_i32)
            )
            var _do_pf = t32_arg + Int32(_PF_DIST) < num_tiles_i32
            var _pf_t = t + _PF_DIST

            var k_slot = k_ring.tile[Self._K_SLOT_ROWS, Self._K_SUB_COLS](
                slot, 0
            )
            var v_slot = v_ring.tile[Self._V_SLOT_ROWS, Self._V_SUB_COLS](
                v_slot_idx, 0
            )
            # CADENCE: the NEXT tile's K LDS slot (`oslot` = the OTHER K
            # double-buffer at depth-2/pf-1). Its first 2 frags are `ds_read`
            # into the PV-tail/QK-tail (the C_PV/C_QK streams below) and handed
            # to the next call's QK seed. Slots 0,1 are nope frags read
            # directly off the slot view (no rope sub-view), so the
            # `load_K_frag[0/1]` reads are valid.
            comptime assert (
                _PF_NEXT_K <= Self._Core._NUM_K_NOPE_TILES
            ), "next-K prefetch reads nope frags only"
            # Pure comptime pointer-offset view (no instructions); the source
            # of the relocated PV-tail / QK-tail next-tile K `ds_read`s below.
            var next_k_slot = k_ring.tile[Self._K_SLOT_ROWS, Self._K_SUB_COLS](
                oslot, 0
            )

            # ===---------- CLUSTER 1: C_QK_MFMA ----------------------=== #
            # Reference asm L352-411 (waves 0-3): the QK MFMAs across the 4 acc
            # banks (nope d=128 + rope d=64) of the single FP32 `att_block`
            # (v[60:123]), K streamed from `k_slot` (the LDS band) into the
            # K register band, interleaved with the work-split K/V DRAM->LDS
            # DMA issued in C_PV_MFMA's prefetch (asm ~366-396).
            #
            # In-file QK math (this path deliberately avoids the delegated-
            # helper `lgkmcnt(0)` staircase drain wall that chops the schedule
            # into <=4-MFMA basic blocks):
            #   1. zero the FP32 accumulator,
            #   2. load whole K_nope into the band (`_load_k_nope_reg`, a
            #      barrier-free LDS->reg loader),
            #   3. a SINGLE soft `lgkmcnt(4)` drain (NOT per-MFMA), then the
            #      nope MFMA loop over `_NUM_K_NOPE_TILES` K-columns (hand-
            #      looped, since `mma_QK` asserts K.K==Q.K but Q carries 3
            #      K-cols = nope 0,1 + rope 2, so the nope phase indexes only
            #      `q[m, 0.._N_NOPE)`),
            #   4. load whole K_rope, soft drain, then the rope MFMA via the
            #      barrier-free `_qk_with_kreg_mla_rope_fp32` primitive (it
            #      skips the post-QK scale under `_FP32_SOFTMAX_SCORES`, leaving
            #      `att_block` RAW for the scale-folded softmax).
            comptime _N_NOPE = Self._Core._NUM_K_NOPE_TILES  # 2 for FP8
            comptime _N_ROPE = Self._Core._NUM_K_ROPE_TILES  # 1 for FP8
            comptime _ATT_H = Self._MmaOp.ATT_LAYOUT.static_shape[0]  # 4
            comptime _ATT_W = Self._MmaOp.ATT_LAYOUT.static_shape[1]  # 1
            comptime _FE = Self._MmaOp.FRAG_ELTS  # 32 FP8 elts = 8 VGPR

            # =========================================================== #
            # STREAMED 32-VGPR K BAND — the reference `v[28:59]` rotating ring.
            # =========================================================== #
            # The reference holds K in a 4-slot × 8-VGPR ring (32 VGPR) and
            # refills each slot ~4 MFMAs AHEAD of its next consumer, so the
            # soft `lgkmcnt(4)` (4 outstanding LDS reads) has slack and the WAR
            # never forces `lgkmcnt(0)` (ref asm QK L355-410: `v[28:35]`
            # consumed by MFMA#1, refilled after MFMA#1, next read at MFMA#5).
            # Contrast a bulk-load (`_load_k_nope` whole 64-VGPR tile +
            # `_load_k_rope` 32-VGPR tile = 96 VGPR resident, then 12 MFMAs
            # re-read a ~3-deep window → each refill overwrites the range an
            # adjacent MFMA just read → WAR → 22 hard `lgkmcnt(0)`). Streaming
            # a 4-slot ring (a) drops the K live-set 96 → 32 VGPR, and (b)
            # breaks the WAR by the 4-ahead fill distance.
            #
            # The ring slots are written by DIRECT comptime slot index on
            # a top-level `reg_alloc` (`ring_v[slot, 0, 0] = frag`), NOT a
            # strided sub-view passed to `load_K` — the latter lands on the
            # wrong VGPRs at non-zero offset. Each fragment is loaded as a SIMD
            # value by `MlaMmaOp.load_K_frag` and fed to `gpu_mma` directly.
            #
            # Per-acc-row K-col schedule (3 cols / acc bank, matching the
            # reference acc-bank-every-3-MFMA structure): nope kk=0 (Q-col
            # 0), nope kk=1 (Q-col 1), rope (Q-col `_N_NOPE`). 4 acc rows ×
            # 3 cols = 12 fragments streamed through the 4-slot ring.
            comptime _NUM_KCOL = _N_NOPE + _N_ROPE  # 3
            comptime _NUM_KFRAG = _ATT_H * _NUM_KCOL  # 12
            comptime _RING = get_defined_int[
                "qk_ring", 4
            ]()  # reference band depth (4 × 8 VGPR = 32 VGPR); sweepable
            comptime _AHEAD = _RING  # refill one full cycle ahead

            # Flat fragment `i` → (acc-row, K-col, is_rope, SMEM sub_id,
            # Q-col). Plain `def`s — called only in `comptime` expressions
            # below, so they fold to immediates at type-check time.
            def _frag_n(i: Int) -> Int:
                return i // _NUM_KCOL

            def _frag_kc(i: Int) -> Int:
                return i % _NUM_KCOL

            def _is_rope(i: Int) -> Bool:
                return _frag_kc(i) >= _N_NOPE

            def _qcol(i: Int) -> Int:
                return _frag_kc(i)

            # SMEM sub_id (load_K linearization `row*width_st + col`):
            #   nope tile [4,2]: width_st=2 → sub_id = n*2 + kk
            #   rope tile [4,1]: width_st=1 → sub_id = n  (in the rope view)
            def _sub_id(i: Int) -> Int:
                return _frag_n(i) if _is_rope(i) else (
                    _frag_n(i) * _N_NOPE + _frag_kc(i)
                )

            _ = att_block.fill(0)

            # The rope K sub-view: the pure ptr-offset slice
            # `_load_k_rope_reg` uses (no data motion). nope frags read
            # `k_slot` directly. `load_K_frag` recomputes the per-lane
            # swizzled bases from each tile's `.ptr`; LLVM CSEs them across
            # the unrolled stream to the reference single base-register pair.
            var k_rope_smem = k_slot.tile[
                Self._Core._K_ROPE_SLOT_ROWS, Self._MmaOp.K_SUB_COLS
            ](Self._Core._K_ROPE_TILE_IDX, 0)

            var _att_v = att_block.vectorize[1, 1, 16]()
            var _q_v = q_reg.vectorize[1, 1, _FE]()

            # The 4 ring slots as SEPARATE SIMD values (NOT one reg_alloc
            # tile — a `[_RING,1,FE]` tile is COALESCED by RA into one 8-VGPR
            # range, re-creating the WAR). Four independent SSA values whose
            # lifetimes overlap (the 4-ahead fill keeps all 4 live at once)
            # force RA to hold 4 distinct 8-VGPR ranges = the 32-VGPR
            # `v[28:59]` band.
            comptime _FRAG = SIMD[Self.config.dtype, _FE]
            var f0 = _FRAG(0)
            var f1 = _FRAG(0)
            var f2 = _FRAG(0)
            var f3 = _FRAG(0)

            # Load fragment `i` from SMEM (the rope sub-view for rope cols).
            @__parameter
            @always_inline
            def _load_frag[i: Int]() -> _FRAG:
                comptime if _is_rope(i):
                    return Self._MmaOp.load_K_frag[_sub_id(i)](k_rope_smem)
                else:
                    return Self._MmaOp.load_K_frag[_sub_id(i)](k_slot)

            # Write fragment value into ring slot `s` (comptime dispatch to
            # one of the 4 distinct SSA values).
            @__parameter
            @always_inline
            def _put[s: Int](var v: _FRAG):
                comptime if s == 0:
                    f0 = v
                elif s == 1:
                    f1 = v
                elif s == 2:
                    f2 = v
                else:
                    f3 = v

            # Read ring slot `s`.
            @__parameter
            @always_inline
            def _get[s: Int]() -> _FRAG:
                comptime if s == 0:
                    return f0
                elif s == 1:
                    return f1
                elif s == 2:
                    return f2
                else:
                    return f3

            # PROLOGUE: prime the first `_RING` slots (no MFMA yet).
            # CADENCE: for the first `_PF_NEXT_K` slots, consume the K fragment
            # the PRIOR tile's PV-tail/QK-tail already read into the
            # closure-scope `_nxt_kf*` hand-off, instead of re-issuing the
            # `ds_read` here. The seed MFMA (i==0 below) then consumes a read
            # issued a full PV cluster earlier -> soft `lgkmcnt(4)` (kills the
            # QK-head drain). This is an UNCONDITIONAL SSA consume — no runtime
            # fresh-load arm. The hand-off is PRIMED from K slot 0 before the
            # loop (tile 0's K), so RA cannot fall back to a fresh `ds_read`;
            # the only source for the seed's K is the closure-scope SSA
            # `_nxt_kf*`, which forces RA to hold it across the PV->QK seam.
            # A runtime-select variant does NOT work here — LLVM
            # PHI-rematerializes the cheap invariant load at the control-flow
            # merge.
            comptime for i in range(_RING):
                comptime if i < _PF_NEXT_K:
                    comptime if i == 0:
                        _put[0](rebind[_FRAG](_nxt_kf0))
                    else:
                        _put[i % _RING](rebind[_FRAG](_nxt_kf1))
                else:
                    _put[i % _RING](_load_frag[i]())

            # SPREAD: emit ONE future-tile DMA sub-call assigned to QK
            # MFMA iteration `i_mfma`, mask-0-fenced so IGLP keeps the
            # `buffer_load...lds` in that MFMA's shadow (ref asm L366-396).
            # The K producer (lower waves) decomposes into 4 sub-calls
            # (`dma_nope`/`dma_rope` x the w_remap / w_remap+4 warp-halves);
            # the V producer (upper waves) into 2 (`_dma_v` x the two
            # halves). The sub-calls are scattered onto distinct MFMA iters
            # by `_PF_K_AT` / `_PF_V_AT` so each load lands after a different
            # MFMA. `_do_pf` (a next tile exists) gates the whole spread; the
            # built K-pair / addr-math is shared (LLVM CSEs the single
            # buffer-resource base) across the 4 K sub-calls.
            comptime _PF_K_AT = (2, 4, 6, 8)  # K sub-call -> MFMA iter
            comptime _PF_V_AT = (2, 6)  # V sub-call -> MFMA iter

            @__parameter
            @always_inline
            def _pf_spread_step[i_mfma: Int]():
                comptime if is_upper:
                    # V producer: 2 halves at _PF_V_AT.
                    comptime for vj in range(len(_PF_V_AT)):
                        comptime if rebind[Int](_PF_V_AT[vj]) == i_mfma:
                            _sched_barrier_zero()
                            Self._Core._dma_v[v_full_v227=Self._V_FULL_V227,](
                                v_ring.tile[
                                    Self._V_SLOT_ROWS, Self._V_SUB_COLS
                                ](v_oslot, 0),
                                v_op,
                                batch_idx_u32,
                                kv_head_idx_u32,
                                _pf_t,
                                w_remap + 4 * vj,
                                l_id,
                            )
                            _sched_barrier_zero()
                else:
                    # K producer: dma_nope/dma_rope x 2 warp-halves at
                    # _PF_K_AT (sub-call kj: half = kj // 2, rope = kj & 1).
                    comptime for kj in range(len(_PF_K_AT)):
                        comptime if rebind[Int](_PF_K_AT[kj]) == i_mfma:
                            _sched_barrier_zero()
                            var _kp = _MlaKDmaPair[Self.config](
                                k_op, batch_idx_u32, kv_head_idx_u32, _pf_t
                            )
                            var _ks = k_ring.tile[
                                Self._K_SLOT_ROWS, Self._K_SUB_COLS
                            ](oslot, 0)
                            # warp-half offset is RUNTIME (`w_remap = w_id&3`);
                            # only the comptime sub-call index `kj` selects the
                            # half (`kj // 2`) and nope-vs-rope (`kj & 1`).
                            var _kh = w_remap + 4 * (kj // 2)
                            comptime if (kj & 1) == 0:
                                _kp.dma_nope(_ks, _kh, l_id)
                            else:
                                _kp.dma_rope(_ks, _kh, l_id)
                            _sched_barrier_zero()

            # STREAM: MFMA[i] consumes slot i%_RING (DISTINCT from the slot
            # refilled `_AHEAD` later), then refills the slot it frees with
            # fragment i+_AHEAD. The 4-ahead fill keeps all 4 slots live, so
            # a refill writes a slot whose previous consumer already issued
            # (next read 4 MFMAs out) — the WAR slack the reference
            # `lgkmcnt(4)` needs (vs a bulk-load / coalesced ring forcing
            # `lgkmcnt(0)`).
            comptime _QK_DRAIN = 4
            # Reference-faithful QK K-band cadence: pin the 4-ahead load/MFMA
            # interleave with the mask-0 reorder fence the ping-pong matmul
            # `_bind` loop uses BETWEEN every op
            # (`amd_ping_pong_matmul.mojo:556-557` -> `schedule_barrier()`).
            # Fencing only at the cluster boundary leaves the per-MFMA refill
            # free for IGLP to sink adjacent to its consumer -> WAR ->
            # `lgkmcnt(0)` + RA-coalesce of the ring to 1 slot. With the fence
            # interspersed, the refill `_put` (a `ds_read`) is pinned a full
            # region AWAY from the MFMA that would WAR-overwrite its slot, so
            # the 4 distinct slots stay live (the matmul fence MECHANISM keeps
            # a hand-specified interleave program-order cannot). The mask-0
            # `schedule_barrier` is the ONE lever IGLP does not override.
            # Reference 10+2 QK split (asm L2245): the reference issues 10 QK
            # MFMAs, then the C_QK->C_V_PREFETCH barrier, then the last 2 QK
            # MFMAs interleaved with the 12 V transpose-reads. We reproduce the
            # split by emitting that barrier (ref L411 == L2245) after the 10th
            # QK MFMA instead of after the 12th, so the trailing 2 QK MFMAs land
            # in the C_V_PREFETCH cluster (the post-loop barrier below is
            # suppressed so the 7-barrier-per-tile count is unchanged). In our
            # IGLP-fused form this surfaces as the PV+nextQK section splitting
            # 18+2 (vs the un-split 20) -- exactly the reference `8 PV + 10
            # nextQK` block plus the 2-QK tail. The split-tail MFMAs land the
            # trailing 2 QK MFMAs in C_V_PREFETCH so they overlap the cross-warp
            # V-sync wait that the 20-fused block serialized behind.
            comptime _QK_SPLIT_AT = 10
            comptime for i in range(_NUM_KFRAG):
                comptime slot = i % _RING
                comptime n = _frag_n(i)
                s_waitcnt[lgkmcnt=UInt32(_QK_DRAIN)]()
                _sched_barrier_zero()  # fence: MFMA region opens here
                var _d = _att_v[n, 0, 0]
                gpu_mma(_d, _get[slot](), _q_v[0, _qcol(i), 0], _d)
                _att_v[n, 0, 0] = _d
                _sched_barrier_zero()  # fence: refill region opens here
                comptime if i == _QK_SPLIT_AT - 1:
                    # The 10+2 split barrier (ref L2245 == the C_QK end /
                    # V-read start). Emitted here so the 2 trailing QK MFMAs
                    # land in C_V_PREFETCH.
                    _sched_barrier_zero()
                    _s_barrier_raw()
                    _sched_barrier_zero()
                comptime if i + _AHEAD < _NUM_KFRAG:
                    _put[(i + _AHEAD) % _RING](_load_frag[i + _AHEAD]())
                # QK-TAIL soften: UNCONDITIONAL next-tile K read (col 1)
                # into the refill region of the 2nd-to-last QK MFMA, so the
                # last QK MFMA's current-tile K operand is no longer the
                # freshest read. Reads `next_k_slot` (in-bounds LDS, `oslot`).
                # No `if _do_pf` => no PHI merge drain. The read must feed a
                # GENUINELY LIVE value, else DCE removes it (the dead-store
                # trap: writing `_nxt_kf0`, which the PV loop overwrites before
                # the next seed, was DCE'd -> no read -> no soften). So the
                # QK-tail writes `_nxt_kf1` (seed col 1), consumed by the next
                # seed; the PV-tail writes `_nxt_kf0` (col 0). Both are live
                # across the seam, both soften their respective tail.
                comptime if (
                    i >= _NUM_KFRAG - 1 - _PF_QK_TAIL and i < _NUM_KFRAG - 1
                ):
                    _sched_barrier_zero()
                    _nxt_kf1 = rebind[_PF_FRAG](
                        Self._MmaOp.load_K_frag[1](next_k_slot)
                    )
                    _sched_barrier_zero()
                # Reference-faithful future-tile prefetch: issue the work-split
                # K[t+1]/V[t+1] DRAM->LDS prefetch INSIDE the QK MFMA stream,
                # SPREAD one sub-call per QK MFMA across the C_QK cluster
                # (ref asm L366-396: one `buffer_load...lds` after each QK
                # MFMA, in the MFMA shadow) via `_pf_spread_step[i]`. The
                # producer is split into per-sub-call steps scattered onto
                # distinct MFMA iters by `_PF_K_AT` / `_PF_V_AT`, each mask-0
                # fenced so IGLP keeps each load in its assigned MFMA shadow.
                # (A single-issue site emitting the WHOLE producer at i==0
                # clumps all 6 `buffer_load...lds` + ~18 SALU addr-math after
                # MFMA #1 — a scc-guarded clump the MFMA cluster cannot fully
                # shadow.) Spreading the loads helps most on the persistent
                # grid, which runs work-items back-to-back so the per-tile
                # clump overhead compounds; the static grid is CU-starved so
                # the clump is mostly hidden there. The work-split (`if
                # stagger` K-vs-V inside `_pf_spread_step`) and `_pf_t` are
                # unchanged; the K half prefetches into K's `oslot`, the V half
                # into V's `v_oslot` (decoupled rings).
                if _do_pf:
                    _pf_spread_step[i]()

            # Ref L411 (end QK MFMA / start V read) is emitted INSIDE the loop
            # above after the 10th QK MFMA (the 10+2 split), so it is not
            # re-emitted here -- this keeps the 7-barrier-per-tile count exact.

            # ===---------- CLUSTER 2: C_V_PREFETCH -------------------=== #
            # Reference asm L411-429: the V transpose-on-read LDS reads
            # (`ds_read_b64_tr_b8`). The reference never materializes V as a
            # whole 64-VGPR tile — it transpose-reads it fragment-at-a-time
            # through the band K vacated (disjoint lifetimes: K is consumed in
            # QK before softmax, V in PV after). So here we only precompute the
            # per-lane V base (loop-invariant; the reference `v227` single-base
            # pattern, hoisted by LLVM); the fragments are streamed JIT in
            # C_PV_MFMA via `load_V_frag`. This is the lean ~210-VGPR layout
            # (no whole-tile V held across the softmax peak). The cross-warp
            # V LDS-write sync (the barriers below + the `_s_barrier_raw`
            # bracket) is PRESERVED — V is written cooperatively to LDS by
            # the work-split (waves 4-7), so the fragment reads still need
            # that sync (a dropped V sync corrupts).
            var v_lane_base = Self._MmaOp.precompute_v_lane_base[
                v_full_v227=Self._V_FULL_V227,
                v227_layout=Self._v227_layout,
            ](v_slot.ptr)

            # ----------------------------------------------------------------
            # V band machinery (slots + helpers) hoisted HERE (out of
            # C_PV_MFMA) so the OPTIONAL `-D v_qktail` QK-tail prefetch
            # below can prime the first `_VAHEAD` V fragments in C_V_PREFETCH
            # and HOLD them across the softmax clusters — matching the
            # reference split (the QK-tail block carries 12 of the 32 V
            # `ds_read_b64_tr_b8` = 3 fragments × 4 reads; ref asm QK-tail
            # L2245-2272). Default (`_V_QKTAIL=False`) the prologue stays in
            # C_PV_MFMA below, so the hoist is SSA-neutral (the band vars are
            # dead until C_PV; the `@__parameter @always_inline` helpers emit
            # nothing until called). The hoist is the lean-V design's mirror:
            # lean-V reads ALL 32 V in C_PV; `v_qktail` moves 12 up to match
            # the reference 12/20 placement, at the cost of holding 24 VGPR
            # (3 frags × 8) across the softmax peak (att 64 + o 64 + Q 24 +
            # softmax state). This is REGISTER-BUDGET-sensitive — the reference
            # absorbs it because the 3 frags land in the K/V SHARED band the QK
            # MFMAs already vacated; whether our band reuses the freed K regs
            # the same way is the gate.
            comptime _V_QKTAIL = get_defined_bool["v_qktail", not _PERSISTENT]()
            comptime _O_FRAG = Self._O_LAYOUT_T.static_shape[2]
            comptime _N_DEPTH = Self.DEPTH // 32
            comptime _NUM_VFRAG = Self._NUM_PV_SUBTILES * _N_DEPTH
            comptime _VFRAG = SIMD[Self.config.dtype, Self._PV_A_FRAG]
            comptime _VRING = 4  # reference 4-slot rotating V band (v[28:59])
            comptime _VAHEAD = 3  # 3 slots in flight (see C_PV_MFMA notes)

            @__parameter
            @always_inline
            def _vstrip(i: Int) -> Int:
                return i // _N_DEPTH

            @__parameter
            @always_inline
            def _vdepth(i: Int) -> Int:
                return i % _N_DEPTH

            # Load V fragment `i` from the per-lane V LDS base (4
            # `ds_read_tr8_b64` joined to one SIMD; `v_lane_base` CSEs to a
            # single base across the unrolled stream — the reference `v227`).
            @__parameter
            @always_inline
            def _vload[i: Int]() -> _VFRAG:
                return rebind[_VFRAG](
                    Self._MmaOp.load_V_frag[
                        _vstrip(i),
                        _vdepth(i),
                        v_full_v227=Self._V_FULL_V227,
                    ](v_lane_base)
                )

            # The 4 ring slots as SEPARATE SSA SIMD values (the QK ring
            # `f0..f3` form: distinct overlapping lifetimes force RA to hold
            # distinct 8-VGPR ranges in the `v[28:59]` band, NOT a single
            # RA-coalesced slot).
            var vf0 = _VFRAG(0)
            var vf1 = _VFRAG(0)
            var vf2 = _VFRAG(0)
            var vf3 = _VFRAG(0)

            @__parameter
            @always_inline
            def _vput[s: Int](var v: _VFRAG):
                comptime if s == 0:
                    vf0 = v
                elif s == 1:
                    vf1 = v
                elif s == 2:
                    vf2 = v
                else:
                    vf3 = v

            @__parameter
            @always_inline
            def _vget[s: Int]() -> _VFRAG:
                comptime if s == 0:
                    return vf0
                elif s == 1:
                    return vf1
                elif s == 2:
                    return vf2
                else:
                    return vf3

            # QK-TAIL V PREFETCH (reference 12/20 split, OPT-IN). Prime the
            # first `_VAHEAD` V fragments HERE (C_V_PREFETCH, ref asm L411-429),
            # so they are HELD across the softmax clusters and consumed in C_PV.
            # This is the reference `ds_read_b64_tr_b8 v227 ... v[28:35]/
            # v[36:43]/v[44:51]` block at ref asm L2245-2272 (3 frags
            # interleaved with the 2 trailing QK MFMAs, before the
            # softmax-entry barrier). The mask-0 fence brackets keep the reads
            # pinned in this cluster.
            comptime if _V_QKTAIL:
                comptime for i in range(min(_VAHEAD, _NUM_VFRAG)):
                    _sched_barrier_zero()
                    _vput[i % _VRING](_vload[i]())
                    _sched_barrier_zero()

            _sched_barrier_zero()
            _s_barrier_raw()  # ref L429 (after final V ds_read)
            _sched_barrier_zero()
            _s_setprio[Int16(0)]()  # drop priority — softmax is compute-light
            _sched_barrier_zero()
            _s_barrier_raw()  # ref L431 (LDS-write drain before softmax)
            _sched_barrier_zero()

            # ===---------- CLUSTER 3: C_SOFTMAX_MAX ------------------=== #
            # Reference asm L431-504: the causal/null mask, then the row-max
            # reduce (the `v_max3_f32` chain + `v_permlane32_swap_b32`
            # cross-half combine) and the fused scale + max-shift (the
            # `v_pk_fma_f32` that folds `scale*log2(e)` into the subtract).
            #
            # In-file softmax max-phase via the BARRIER-FREE `OnlineSoftmax`
            # recurrence primitives (NOT `_softmax_tile_fp32`, which bundles
            # an IGLP `sched_barrier_exp_pairs` fence + is the body delegation
            # this loop avoids). `att_block` arrives RAW (un-scaled QK — the
            # rope helper skipped the post-QK scale), so the scale folds into
            # the max-subtract here:
            #   - t==0 (no prior running max): `seed_tile0_scaled` =
            #     `max_vec = log2_scale*col_max(raw)` (the `v_max3` chain via
            #     `_col_max_scalar_v3max`) + `att = log2_scale*att - max_vec`
            #     (the fused `v_pk_fma`).
            #   - t>0: `col_max_acc_scaled` = `max_vec = max(max_prev,
            #     log2_scale*col_max(raw))` (running max), then
            #     `sub_max_scaled` = the fused `v_pk_fma` shift. The α-rescale
            #     of o_reg/L is deferred to C_EXP (it commutes — it touches
            #     o_reg/norm_vec, not att_block — so the ref L504 cluster
            #     boundary falls cleanly between the shift and the exp).
            #
            # Mask FIRST so the -INF sentinel on out-of-range keys is seen by
            # the row-max (the NullMask path applies no sentinel; the Causal
            # path does — both must precede col_max).
            mask.apply(
                att_block,
                tile_idx,
                t,
                start_pos,
                UInt32(head_idx),
                batch_idx_u32,
                l_id,
            )
            if t > 0:
                softmax.col_max_acc_scaled(att_block, scale_log2e)
                softmax.sub_max_scaled(att_block, scale_log2e)
            else:
                softmax.seed_tile0_scaled(att_block, scale_log2e)
            _sched_barrier_zero()
            _s_barrier_raw()  # ref L504 (between pk_fma and v_exp)
            _sched_barrier_zero()

            # ===---------- CLUSTER 4: C_EXP + EAGER_RESCALE ----------=== #
            # Reference asm L504-707: the `v_exp_f32` over all 64 acc entries,
            # then (t>0) the α = exp2(m_prev - m_new) rescale factor, the
            # L_running update (L = α*L + ΣP), and the EAGER acc rescale
            # `o_reg *= α` BEFORE PV (distinct from the LAZY post-PV rescale
            # of the MLA-decode path).
            #
            # In-file, via the BARRIER-FREE `OnlineSoftmax` recurrence
            # primitives — the EXACT math sequence of `MlaPrefillV2Core`'s
            # `_softmax_tile_fp32` t>0 body (in `mla_components.mojo`)
            # minus its IGLP fence:
            #   - `update_scale_unconditional`: `scale_vec =
            #     math_exp2(max_prev - max_new)`, then `max_prev = max_new`.
            #     This IS the α-guard — for a masked row whose new max landed
            #     on the sentinel, `exp2(huge_negative)` does NOT flush to 0
            #     on gfx950, but the EAGER path applies the SAME α
            #     consistently to o_reg AND L,
            #     so the recurrence stays exact (no stale lazy-skip scale —
            #     the lazy bug needs the skip branch, which the eager path
            #     never takes).
            #   - `rescale_output(o_reg)`: EAGER `o_reg *= α` (the reference
            #     pre-PV acc rescale, asm `v_mul_f32 v[124..187]`).
            #   - `apply_unconditional_norm_rescale`: `norm_vec *= α`
            #     (the reference `v_mul_f32 v198`), correcting the running
            #     denominator before this tile's mass is summed in.
            #   - `exp2_inplace_range[0, _ATT_PER_LANE]`: the 64 `v_exp_f32`
            #     over the whole score tile (P[t] = exp2(scaled QK - max)).
            #   - `col_sum_acc`: `norm_vec += ΣP[t]` (the reference 64
            #     `v_add_f32` into v198).
            # For t==0 the seed already did the max-subtract; just exp + sum.
            comptime _APL = Self._Core._ATT_PER_LANE
            if t > 0:
                softmax.update_scale_unconditional()
                softmax.rescale_output(o_reg)
                softmax.apply_unconditional_norm_rescale()
            Self._MmaOp.exp2_inplace_range[0, _APL](att_block)
            softmax.col_sum_acc(att_block)
            _sched_barrier_zero()
            _s_barrier_raw()  # ref L707 (end acc-rescale + L-update)
            _sched_barrier_zero()

            # ===---------- CLUSTER 5: C_FP8_PACK ---------------------=== #
            # Reference asm L707-740: the in-place FP32 -> FP8 pack of P
            # (`v_cvt_pk_fp8_f32 ... op_sel:[0,0,1]` telescope) into
            # `att_block`'s OWN low quarter. FILL via the BARRIER-FREE
            # `Self._Core._qk_collapse_inplace` (the FP8 cast, an authorized
            # low-level primitive — it carries no drain/barrier; it is the
            # `_att_bf16_full` telescope over an FP8 view aliasing
            # `att_block.ptr`, so no separate FP8 P tile is materialized).
            # `p_view` aliases `att_block`'s storage and is HELD live for
            # C_PV_MFMA.
            var p_view = Self._Core._qk_collapse_inplace(att_block)
            _sched_barrier_zero()
            _s_barrier_raw()  # ref L740 (end FP8 P-cast)
            _sched_barrier_zero()
            # Ref L741-743: soft drain + raise priority for the PV MFMAs.
            s_waitcnt[vmcnt=UInt32(0), lgkmcnt=UInt32(8)]()
            _s_setprio[Int16(1)]()
            _sched_barrier_zero()
            _s_barrier_raw()  # ref L744 (drain before PV MFMA)
            _sched_barrier_zero()

            # ===---------- CLUSTER 6: C_PV_MFMA ----------------------=== #
            # Reference asm L744-787: the PV MFMAs (32x32x64 FP8 -> FP32) into
            # the 4 O acc banks (v[124:187]) -- `o_reg += V[t]^T @ P[t]`.
            #
            # Reference-faithful non-materialized V streamed through a 4-slot
            # rotating band `v[28:59]` (the V-side twin of the QK 4-ahead K
            # ring). Each V fragment is transpose-read JIT via `load_V_frag`
            # into a ring slot `_VAHEAD` MFMAs AHEAD of its consuming PV MFMA,
            # so several slots' reads are outstanding when the MFMA waits ->
            # WAR slack -> the per-MFMA drain is SOFT `lgkmcnt(8)` (ref asm
            # L744-785: 4 `ds_read_b64_tr_b8` then 1 `v_mfma`, repeat) instead
            # of a 1-slot band's hard `lgkmcnt(0)`. The ring slots are DISTINCT
            # SSA SIMD values (`_vget`/`_vput` comptime dispatch), NOT a
            # `reg_alloc[N,1,FE]` tile (RA coalesces an in-order-consumed
            # tile ring back to 1 slot -> WAR re-forms). Each (consume,
            # refill) pair is bracketed by the mask-0 `_sched_barrier_zero()`
            # reorder fence -- the ONE lever that constrains IGLP itself
            # (pure program order + `s_waitcnt` are freely re-clustered) --
            # so the refill `ds_read` cannot sink adjacent to the MFMA that
            # WARs its slot. `gpu_mma` is called DIRECTLY on the ring-slot
            # SIMD value (the single-fragment reduction of `mma_PV`: `o[_n]
            # += v_slot^T @ p[_pv]`, O_h=O_w=K_count=1). WITHIN-TILE only.
            #
            # The band slots (`vf0..vf3`), helpers (`_vstrip`/`_vdepth`/
            # `_vload`/`_vput`/`_vget`) and the `_VRING`/`_VAHEAD`/`_N_DEPTH`/
            # `_NUM_VFRAG`/`_O_FRAG`/`_VFRAG` constants are declared in
            # C_V_PREFETCH above (hoisted so the optional `v_qktail`
            # prefetch can prime + hold the first `_VAHEAD` fragments across
            # softmax). `_VAHEAD=3` slots in flight: a 2-in-flight `_VAHEAD=2`
            # RA-coalesces back to 2 physical regs -> WAR -> `lgkmcnt(4)`; the
            # per-PV-MFMA drain lands at `lgkmcnt(8)` matching the reference (a
            # 4-in-flight `_VAHEAD=4` over-deepens to `lgkmcnt(12)`). Fragment
            # `i` -> (`_pv = i // _N_DEPTH` strip, `_n = i % _N_DEPTH`
            # depth/O-bank): `_o_bank = o_reg.tile(_n)`, `_p_sub =
            # p_view.tile(_pv)` — the 1:1 band-slot <-> O-bank map.

            # PROLOGUE: prime the first `_VAHEAD` slots (no MFMA yet) so
            # `_VAHEAD` ring slots are simultaneously live -> RA holds that
            # many distinct 8-VGPR ranges (the no-coalesce condition) and the
            # steady-state refill writes a slot `_VAHEAD` MFMAs ahead of its
            # consumer. Under `v_qktail` this prologue already ran in
            # C_V_PREFETCH (the 12-V QK-tail prefetch) and `vf0..vf2` are HELD
            # live across softmax — so it is suppressed here to avoid a
            # double-read.
            comptime if not _V_QKTAIL:
                comptime for i in range(min(_VAHEAD, _NUM_VFRAG)):
                    _sched_barrier_zero()
                    _vput[i % _VRING](_vload[i]())
                    _sched_barrier_zero()

            # STREAM: MFMA[i] consumes slot i%_VRING (read `_VAHEAD`
            # MFMAs ago), then refills the freed slot with fragment
            # i+_VAHEAD. The fence brackets the (consume, refill) so
            # IGLP cannot sink the refill `ds_read` adjacent to the
            # MFMA that WARs its slot.
            comptime for i in range(_NUM_VFRAG):
                comptime slot = i % _VRING
                comptime _pv = _vstrip(i)
                comptime _n = _vdepth(i)
                var _p_sub = p_view.tile[1, 1, Self._PV_A_FRAG](_pv, 0, 0)
                var _p_v = _p_sub.vectorize[1, 1, Self._PV_A_FRAG]()
                var _o_bank = o_reg.tile[1, 1, _O_FRAG](_n, 0, 0)
                var _o_v = _o_bank.vectorize[1, 1, _O_FRAG]()
                _sched_barrier_zero()  # fence: MFMA region opens
                # Single-fragment `mma_PV`: o[_n] += v_slot^T @ p[_pv].
                var _d = _o_v[0, 0, 0]
                gpu_mma(_d, _vget[slot](), _p_v[0, 0, 0], _d)
                _o_v[0, 0, 0] = _d
                _sched_barrier_zero()  # fence: refill region opens
                comptime if i + _VAHEAD < _NUM_VFRAG:
                    _vput[(i + _VAHEAD) % _VRING](_vload[i + _VAHEAD]())

                # CADENCE: relocate the NEXT tile's first `_PF_NEXT_K` K
                # ring-fragment `ds_read`s (LDS->reg) into the last
                # `_PF_NEXT_K` PV MFMAs' shadow — ref L776-785's
                # `ds_read_b128 v226 offset:24960` (the other K db-slot)
                # interleaved among the last PV MFMAs. The read is pinned by
                # the mask-0 fence so IGLP cannot sink it back out of the PV
                # tail. The value is HANDED to the next call's QK prologue via
                # the closure-scope `_nxt_kf*` (consumed at the prologue
                # above). Reads `oslot` = the next tile's K slot, already
                # DMA'd in C_QK + drained `vmcnt(0)` at the FP8 cluster. Slots
                # 0,1 are both nope frags (`_sub_id`=0,1, read `next_k_slot`
                # directly — no rope sub-view for _PF_NEXT_K<=2).
                #
                # PV-TAIL SOFTEN: issue the next-tile K read (col 0)
                # in the refill region of the 2nd-to-last PV MFMA, so the LAST
                # PV MFMA sees its V operand as OLDER than the K read ->
                # `lgkmcnt(0)` softens to `lgkmcnt(K_count)`. A `lgkmcnt(0)`
                # softens ONLY IF the consuming MFMA's operand is NOT the
                # freshest outstanding LDS read; the plain post-last-MFMA
                # placement (the `elif` below) lands the read AFTER the last
                # MFMA and does NOT soften it. This reproduces ref L776-785:
                # `ds_read_b128 v226 offset:24960` issued BETWEEN the
                # penultimate PV MFMAs, last MFMA + next-tile QK seed consume
                # at `lgkmcnt(4)`. The read is UNCONDITIONAL (no `if _do_pf`):
                # a runtime guard PHI-merges the read into a control-flow merge
                # and DRAINS `lgkmcnt(0)` there. Reading `next_k_slot`
                # unconditionally is SAFE — it is an in-bounds LDS slot view
                # (`oslot = (t+1)%ring`); the value reaches the next QK seed
                # only via `_nxt_kf*`, consumed only when a next tile exists.
                # On the last tile `oslot` is stale but the read is never used
                # -> bit-exact, no PHI, no merge drain.
                comptime if (
                    i >= _NUM_VFRAG - 1 - _PF_PV_TAIL and i < _NUM_VFRAG - 1
                ):
                    _sched_barrier_zero()
                    _nxt_kf0 = rebind[_PF_FRAG](
                        Self._MmaOp.load_K_frag[0](next_k_slot)
                    )
                    _sched_barrier_zero()
                # Plain post-last-MFMA placement for col 1 (the LAST PV MFMA).
                # Guarded by `if _do_pf` because it is NOT freshest-before-MFMA
                # (no soften to lose), and the runtime guard avoids reading a
                # stale slot when there is no next tile.
                elif i >= _NUM_VFRAG - _PF_NEXT_K:
                    if _do_pf:
                        _sched_barrier_zero()
                        _nxt_kf1 = rebind[_PF_FRAG](
                            Self._MmaOp.load_K_frag[1](next_k_slot)
                        )
                        _sched_barrier_zero()

            # The work-split K[t+1]/V[t+1] prefetch is issued in the C_QK
            # MFMA stream above (ref L366-396), NOT here in the PV shadow.
            # The reference places the 6 `buffer_load...lds` after the QK
            # MFMAs; the PV-shadow position is not used.
            _s_setprio[Int16(0)]()  # drop priority back (reference per-iter)
            # NB: NO inter-tile `s_barrier` here — the reference PV[t]->QK[t+1]
            # boundary has none; the next sub-iter's C_QK_MFMA opens with
            # its own L411 boundary, and the conserved prologue stagger
            # carries the cross-warp ordering (the cross-warp K-handoff sync is
            # relocated into the QK head, KEPT not removed — dropping it is a
            # corruption edge).

        # ---- Back-edge driver: 2 sub-iters per back-edge --------------
        # The reference main loop processes 2 KV-blocks of 128 per back-edge.
        # Steady-state pairs [t, t+1] while two full tiles remain, then a
        # 1-tile remainder. The barrier cadence is 2 x 7 = 14 `s_barrier` per
        # back-edge. (`num_tiles` is runtime, so this is a runtime
        # steady-state-by-2, not a comptime whole-loop unroll.)
        # PRIME the hand-off from tile 0's K slot 0 so the QK-prologue
        # consume is unconditional (no runtime fresh-load arm). Tile 0's K is
        # DMA'd + drained in the prologue above, so slot 0 holds tile 0's K
        # here — reading frags 0/1 off it is the SAME data tile 0's QK seed
        # would fresh-load (bit-exact). This makes the prologue SSA-consume
        # valid for every tile incl. tile 0.
        var _k0 = k_ring.tile[Self._K_SLOT_ROWS, Self._K_SUB_COLS](0, 0)
        _nxt_kf0 = rebind[_PF_FRAG](Self._MmaOp.load_K_frag[0](_k0))
        _nxt_kf1 = rebind[_PF_FRAG](Self._MmaOp.load_K_frag[1](_k0))

        # Reference wave-half body fork (the `s_cmp_lt_i32 s62, 4` +
        # `s_cbranch` at asm L261): the lower half (waves 0-3) runs the
        # K-streaming body (label_01D6), the upper half (waves 4-7) the
        # V-streaming body (label_073D). Two compute-identical bodies that
        # differ only in the per-tile DMA source (K s[8:11] vs V s[12:15]); the
        # comptime `is_upper` split is INTENDED to make the compiler emit BOTH
        # so the +4 prologue stagger overlaps one half's softmax-VALU under the
        # other half's MFMA. NOTE: in current codegen the split does NOT
        # diverge the body — LLVM keeps a single lockstep stream (no
        # `s_cmp...4` two-stream fork), so the +4 offset is matched for
        # reference fidelity but is perf-inert until a real warp-group fork is
        # built.
        if stagger:
            var t32: Int32 = 0
            while t32 + 1 < max_num_tiles_i32:
                _one_tile_exact[True](t32)
                _one_tile_exact[True](t32 + 1)
                t32 += 2
            if t32 < max_num_tiles_i32:
                _one_tile_exact[True](t32)
        else:
            var t32: Int32 = 0
            while t32 + 1 < max_num_tiles_i32:
                _one_tile_exact[False](t32)
                _one_tile_exact[False](t32 + 1)
                t32 += 2
            if t32 < max_num_tiles_i32:
                _one_tile_exact[False](t32)

        # =================================================================
        # MASKED TAIL + EPILOGUE — remaining reference-structural follow-ons.
        # =================================================================
        # Both are functionally correct here via substitutes; what is
        # deferred is matching the reference exact tail/epilogue STRUCTURE.
        #
        # Masked tail (the reference label_0C18 / label_1336): the reference
        # repeats the QK+softmax+PV cadence with a causal mask (`v_cmp_lt_i32`
        # + `v_cndmask` -INF). Here the causal cap on `max_num_tiles_local`
        # (computed in `_run_one_work`) bounds the loop instead -- correct; the
        # dedicated masked-tail cluster is the structural follow-on.
        #
        # Epilogue (the reference label_1A51, asm L4956): the reference does a
        # 4-barrier resync + final L-reduce / rcp / BF16 cast / output store.
        # Here we use the verified barrier-free `normalize_output` +
        # `_store_o_to_gmem` primitives (correct output layout); matching the
        # reference 4-barrier resync is the structural follow-on.
        softmax.normalize_output(o_reg)

        comptime _o_view_layout = Self._MmaOp.O_T_LAYOUT
        var o_normalized_view = TileTensor[
            .float32,
            type_of(_o_view_layout),
            MutUntrackedOrigin,
            address_space=.LOCAL,
        ](o_reg.ptr, _o_view_layout)
        var epilogue_writer = RegTileEpilogue[out_dtype, 1](o_warp_2d)
        # int32 clamp: `seq_len`/`block_tile_idx`/`w_id` originate as Int32
        # (`q.dim[1]`, `block_idx.y`/`w_qo_start//BM`, `warp_id()`) and the
        # valid-row count is bounded by `BM`(=256). Keeping the `min`/`max`
        # clamp in i32 avoids the i64 vector compare (`v_cmp_*_i64`) gfx950
        # emits for an `Int` (i64) `min`/`max` — there is no signed-i64
        # scalar compare on CDNA4. `BM`/`Q_BLOCK_SIZE` are comptime, so the
        # only runtime operands are the three narrowed values. Boundary back
        # to `Int` for `_store_*`.
        comptime _BM_I32 = Int32(Self.BM)
        comptime _QBS_I32 = Int32(Self.Q_BLOCK_SIZE)
        var seq_len_i32 = Int32(seq_len)
        var block_tile_idx_i32 = Int32(block_tile_idx)
        var w_id_i32 = Int32(w_id)
        var valid_q_rows_i32 = max(
            Int32(0),
            min(_BM_I32, seq_len_i32 - block_tile_idx_i32 * _BM_I32),
        )
        var valid_q_rows_in_warp = Int(
            min(
                _QBS_I32,
                max(Int32(0), valid_q_rows_i32 - w_id_i32 * _QBS_I32),
            )
        )
        Self._Core._store_o_to_gmem[out_dtype](
            o_normalized_view,
            epilogue_writer,
            l_id,
            valid_q_rows_in_warp,
        )

        # ---- LOWER-HALF +4 TAIL COMPENSATION (gated) ------------------
        # The upper half takes `_STAGGER_EXTRA` (=4) extra bare `s_barrier`s
        # at the PROLOGUE (see the prologue stagger keystone above). On gfx950
        # `s_barrier` is a workgroup-wide COUNTING rendezvous (CDNA4 ISA
        # 15250: "force each wavefront to wait until all OTHER wavefronts
        # reach the same instruction"); there is NO split/named barrier on
        # this ISA. So the lower half must pay a MATCHING +4 somewhere or the
        # per-CU totals are UNEQUAL (upper = lower + 4) and the extra upper
        # barriers only survive via the ENDPGM exception. Two disciplines:
        #
        # `exact_stagger` (the EXACT reference replica) emits the lower
        # +4 at the TAIL EVERY work-item (drops the `apply_prologue_stagger`
        # gate), matching the every-work-item upper +4 prologue. This is the
        # reference `label_1A51` (4 bare `s_barrier`s, lower half only, before
        # the `s_branch label_011C` back-edge): the lower half catches up the
        # +4 it ceded to the upper at the prologue, EVERY work-item. Per
        # work-item the totals are EQUAL (no +4N accumulation, no deadlock) AND
        # the skew RE-FORMS each work-item -> a steady phase skew across the
        # CU's whole work stream (the overlap-producing structure). The upper
        # half skips the tail (matching the reference `label_1336/16C6 ->
        # label_1A55`, which jumps PAST `label_1A51`).
        #
        # DEFAULT = `persistent` (the win lives in the persistent grid;
        # static grid is a no-op). Override with `-D exact_stagger=`.
        comptime _EXACT_STAGGER_TAIL = get_defined_bool[
            "exact_stagger", _PERSISTENT
        ]()
        comptime if _EXACT_STAGGER_TAIL:
            # EVERY work-item, lower half only -> the reference `label_1A51`.
            if not stagger:
                comptime for _e in range(_STAGGER_EXTRA):
                    _sched_barrier_zero()
                    _s_barrier_raw()
                    _sched_barrier_zero()

    # ===-------------------------------------------------------------=== #
    # Main kernel body: multi-block 8-warp MLA forward.
    # ===-------------------------------------------------------------=== #
    @__llvm_metadata(`llvm.amdgpu-waves-per-eu`=__mlir_attr.`"1,1"`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name(
        t"mla_prefill_v2_amd_{q_dtype}_{output_dtype}_BM{Self.BM}_KV{Self.KV_BLOCK}_D{Self.DEPTH}"
    )
    @staticmethod
    def run[
        k_nope_t: MHAOperand,
        k_rope_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        q_dtype: DType,
        output_dtype: DType,
        q_layout: TensorLayout,
        o_layout: TensorLayout,
        q_engine: TensorEngine,
        o_engine: TensorEngine,
        ragged: Bool = False,
    ](
        q: TileTensor[q_dtype, q_layout, ImmutAnyOrigin, Engine=q_engine],
        k_nope_op: k_nope_t,
        k_rope_op: k_rope_t,
        v_op: v_t,
        o: TileTensor[output_dtype, o_layout, MutAnyOrigin, Engine=o_engine],
        mask_functor: mask_t,
        scale: Float32,
        num_keys_dev: Int32,
        start_pos_dev: Int32,
        work_indptr_ptr: UnsafePointer[Int32, ImmutAnyOrigin] = UnsafePointer[
            Int32, ImmutAnyOrigin
        ].unsafe_dangling(),
        work_info_ptr: UnsafePointer[Int32, ImmutAnyOrigin] = UnsafePointer[
            Int32, ImmutAnyOrigin
        ].unsafe_dangling(),
        num_works_dev: Int32 = 0,
    ):
        """Multi-block 8-warp MLA forward: reference integrated cadence.

        Grid: `(NUM_HEADS, ceildiv(seq_len, BM), batch)`. Each block owns
        one `(batch, head, BM-tile)` slice; the 8 warps split the
        BM-tile's Q rows. Same grid/operand contract as
        `MlaPrefillV2Core.run`.

        Parameters:
            k_nope_t: Compile-time operand type for the K nope
                segment.
            k_rope_t: Compile-time operand type for the K rope
                segment.
            v_t: Compile-time operand type for the V segment.
            mask_t: Compile-time mask predicate type (causal, null,
                etc.).
            q_dtype: Element type of the Q tensor; must equal
                `config.dtype`.
            output_dtype: Element type of the output tensor; must
                equal `config.output_dtype`.
            q_layout: Memory layout of the Q tile tensor.
            o_layout: Memory layout of the output tile tensor.
            q_engine: Engine of the Q tile tensor.
            o_engine: Engine of the output tile tensor.
            ragged: Whether the batch uses ragged variable-length
                sequences (defaults to `False`).

        Args:
            q: Q tile tensor at d_qk = d_nope + d_rope.
            k_nope_op: K (nope segment) operand, `head_dim_idx=0`. Also
                serves as the single-base K loader source (the
                `_MlaKDmaPair` reads both nope cols [0, D_NOPE) and rope
                cols [ROPE_CACHE_OFFSET, +D_ROPE) from this one operand).
            k_rope_op: K (rope segment) operand (unused here, the
                unified `_MlaKDmaPair` slices rope from `k_nope_op`'s
                full latent-cache row; kept in the signature for
                contract parity with `MlaPrefillV2Core.run` and the
                dispatcher).
            v_op: V operand (= nope segment of the latent cache),
                `head_dim_idx=0`.
            o: Output tile tensor at d_pv = depth.
            mask_functor: Per-tile mask predicate (causal / null / ...).
            scale: Softmax scale (`1/sqrt(d_qk)`).
            num_keys_dev: Runtime length of the K/V sequence.
            start_pos_dev: Position of the first Q row in the global
                sequence.
            work_indptr_ptr: Persistent-prefill work partition
                prefix-sum `[num_cu+1]` (device). Threaded for the S2
                persistent grid; unused by the current static grid.
            work_info_ptr: Persistent-prefill flat `WorkInfo`
                array `[num_works*8]` int32 (device). Threaded for S2;
                unused by the current static grid.
            num_works_dev: Total number of work-tiles in `work_info_ptr`.
                Threaded for S2; unused by the current static grid.
        """
        var num_keys = Int(num_keys_dev)
        var start_pos = Int(start_pos_dev)
        var num_works = Int(num_works_dev)
        comptime assert Self._IS_DSV_MLA_SHAPE, (
            "MlaPrefillV2: only the FP8 / KV>=128 / 32x32x64 shape is"
            " supported in Phase 1 (the reference integrated cadence target)."
        )
        comptime assert (
            q_dtype == Self.config.dtype
        ), "MlaPrefillV2.run: `q.dtype` must equal `config.dtype`"
        comptime assert (
            output_dtype == Self.config.output_dtype
        ), "MlaPrefillV2.run: `o.dtype` must equal `config.output_dtype`"
        _ = k_rope_op
        # S2.1: persistent-prefill work partition is threaded but not yet
        # consumed — the static `(NUM_HEADS, ceildiv(seq, BM), batch)` grid is
        # still active. S2.2 flips the grid to persistent and reads these.
        _ = work_indptr_ptr
        _ = work_info_ptr
        _ = num_works

        var q_typed = rebind[
            TileTensor[
                Self.config.dtype, q_layout, ImmutAnyOrigin, Engine=q_engine
            ]
        ](q)

        var seq_len = Int(readfirstlane(Int32(q.dim[1]())))
        var num_tiles = Int(
            readfirstlane(Int32(ceildiv(num_keys, Self.KV_BLOCK)))
        )
        comptime assert (
            Self.NUM_HEADS % Self.NUM_KV_HEADS == 0
        ), "MlaPrefillV2: NUM_HEADS must be a multiple of NUM_KV_HEADS"

        # ---- R1: reference 160 KB 3-region LDS. ----------------------
        # Two disjoint SMEM regions — one `_RING_DEPTH`-slot K ring (depth-2),
        # one `_V_RING_DEPTH`-slot V ring (depth-4) — with DECOUPLED depths
        # (the reference V region is ~4 slots, K's ~2). The two allocations are
        # the two DISJOINT regions (the reference K@0xc300 / V@0x18600); each
        # ring slot is a `.tile[_K_SLOT_ROWS, _K_SUB_COLS](slot, 0)` /
        # `.tile[_V_SLOT_ROWS, _V_SUB_COLS](v_slot_idx, 0)` sub-view, shape-
        # identical to the per-slot `smem_layout_k` / `smem_layout_v` the
        # shared loaders consume. For FP8 KV=128:
        #   K ring 2*24576 = 49152 B + V ring 4*16384 = 65536 B
        #   + Q slab 49152 B = 163840 B = exactly the reference 160 KB group
        # segment (no slack). The per-lane V base is computed per slot inside
        # the loop (`precompute_v_lane_base`), so no per-slot lane base needs
        # threading through the mainloop signature.
        var k_ring_tt = smem_alloc[Self.config.dtype, alignment=Self._K_ALIGN](
            Self._K_RING_LAYOUT
        )
        var v_ring_tt = smem_alloc[Self.config.dtype, alignment=Self._V_ALIGN](
            Self._V_RING_LAYOUT
        )

        # ---- Reference Q@0x0 LDS region (49920 B). --------------------
        # The resident-Q LDS region: a `[BM, D_QK]` FP8 slab into which Q is
        # staged DRAM->LDS->VGPR once per work-tile via `_load_q_lds_exact`
        # (held in registers across all KV blocks). This is the third of the
        # reference 3 disjoint LDS regions (Q@0x0 / K@0xc300 / V@0x18600).
        var q_lds_tt = smem_alloc[Self.config.dtype, alignment=Self._K_ALIGN](
            Self._Q_LDS_LAYOUT
        )

        var w_id = Int(readfirstlane(warp_id()))
        var l_id = Int(lane_id())

        # Wave-specialization split: lower warp-half `[0, NUM_WARPS/2)` vs
        # upper half `[NUM_WARPS/2, NUM_WARPS)`. The upper half hits the
        # asymmetric +4 prologue stagger barrier in `_attend_exact`,
        # phase-shifting it relative to the lower half. Mirrors
        # `MlaPrefillV2Core`'s `stagger` predicate.
        var stagger = w_id >= (Self.NUM_WARPS // 2)

        comptime _PERSISTENT = get_defined_bool["persistent", False]()
        comptime _GROUP = Self.NUM_HEADS // Self.NUM_KV_HEADS

        # One work-item = `BM` (256) query TOKENS of ONE head (token-major; the
        # reference m32x8 layout — the 256 MMA rows are 256 tokens, NOT 16 tok
        # x 16 head). The static grid and the reference persistent work-loop
        # drive this SAME body; they differ only in how (head_idx,
        # block_tile_idx, batch_idx) are sourced (block_idx vs WorkInfo).
        @__parameter
        @always_inline
        def _run_one_work(
            head_idx: Int,
            block_tile_idx: Int,
            batch_idx: Int,
            is_first: Bool,
        ):
            var kv_head_idx = head_idx // _GROUP
            var tile_idx = block_tile_idx * Self.NUM_WARPS + w_id

            # Causal cap on `max_num_tiles` (mirror MlaPrefillV2Core, including
            # the floor-at-4 correctness fix for the short-tile case).
            # int32 clamp: `block_tile_idx`/`start_pos`/`num_tiles` originate
            # as Int32 and the tile count is bounded (KV>=128, seq<=8192 ->
            # num_tiles<=64). The `<` comparisons in the causal cap + floor4
            # would otherwise lower to i64 vector compares (`v_cmp_*_i64`);
            # there is no signed-i64 scalar compare on gfx950.
            # `NUM_WARPS`/`Q_BLOCK_SIZE`/`KV_BLOCK` are comptime; the result
            # is consumed as `Int32(max_num_tiles_local)` at the loop bound,
            # so computing it in i32 also removes the i64->i32 round-trip.
            comptime _NW_I32 = Int32(Self.NUM_WARPS)
            comptime _QBS_I32 = Int32(Self.Q_BLOCK_SIZE)
            comptime _KVB_I32 = Int32(Self.KV_BLOCK)
            var num_tiles_i32 = Int32(num_tiles)
            var max_tile_idx_local_i32 = (
                Int32(block_tile_idx) * _NW_I32 + _NW_I32 - 1
            )
            var max_q_end_pos_i32 = (
                max_tile_idx_local_i32 + 1
            ) * _QBS_I32 + Int32(start_pos)
            var max_num_tiles_calc_i32 = ceildiv(max_q_end_pos_i32, _KVB_I32)
            var max_num_tiles_local: Int
            comptime if mask_t == CausalMask:
                var capped_i32 = min(max_num_tiles_calc_i32, num_tiles_i32)
                var floor4_i32 = min(Int32(4), num_tiles_i32)
                max_num_tiles_local = Int(max(capped_i32, floor4_i32))
            else:
                max_num_tiles_local = num_tiles

            # Per-(batch, head) 2D views over Q (at d_qk) and O (at d_pv).
            var q_batch_coord = 0 if ragged else batch_idx
            var q_2d = q_typed.tile(
                Coord(Idx[1], Int32(seq_len), Idx[1], Idx[Self.D_QK]),
                Coord(q_batch_coord, Idx[0], head_idx, Idx[0]),
            ).reshape(
                Self._QPerHeadLayoutT(
                    Coord(Int32(seq_len), Idx[Self.D_QK]),
                    Coord(Idx[Self._Core._Q_ROW_STRIDE], Idx[1]),
                )
            )
            var o_2d = o.tile(
                Coord(Idx[1], Int32(seq_len), Idx[1], Idx[Self.DEPTH]),
                Coord(q_batch_coord, Idx[0], head_idx, Idx[0]),
            ).reshape(
                Self._OPerHeadLayoutT(
                    Coord(Int32(seq_len), Idx[Self.DEPTH]),
                    Coord(Idx[Self.NUM_HEADS * Self.DEPTH], Idx[1]),
                )
            )
            var q_warp_block_idx = block_tile_idx * Self.NUM_WARPS + w_id
            var q_warp_2d = q_2d.tile[Self.Q_BLOCK_SIZE, Self.D_QK](
                q_warp_block_idx, 0
            )
            var o_warp_2d = o_2d.tile[Self.Q_BLOCK_SIZE, Self.DEPTH](
                q_warp_block_idx, 0
            )

            var _batch_idx_u32 = UInt32(batch_idx)
            var _kv_head_idx_u32 = UInt32(kv_head_idx)

            # ---- Resident Q (loaded once, held across all KV blocks). ----
            # Reference Q-in-LDS: Q is staged DRAM->LDS->VGPR (resident) via
            # `_load_q_lds_exact`, returning the `_Q_LAYOUT_MLA_T` fragment.
            # FP8: post-QK scale (no prescale).
            var scale_log2e = scale * 1.4426950408889634
            var q_reg = Self._load_q_lds_exact(
                q_warp_2d, q_lds_tt, w_id, scale_log2e
            )

            # ---- Per-work-item register state ----------------------------
            # The reference footprint: Q(24) + att(64 FP32) + o_reg(64 FP32).
            var o_reg = reg_alloc[.float32](Self._MmaOp.O_LAYOUT)
            var softmax = OnlineSoftmax[Self._SOFTMAX_DTYPE]()
            _ = o_reg.fill(0)
            var att_block = reg_alloc[Self._SOFTMAX_DTYPE](
                Self._MmaOp.ATT_LAYOUT
            )

            # ---- Reference-exact inner loop ------------------------------
            # The 6-cluster barrier-delimited mainloop that emits the reference
            # `mla_pfl_qh192_vh128_m32x8_n128x1` instruction schedule directly.
            Self._attend_exact(
                q_reg,
                o_reg,
                softmax,
                mask_functor,
                att_block,
                k_ring_tt,
                v_ring_tt,
                k_nope_op,
                v_op,
                o_warp_2d,
                num_tiles,
                max_num_tiles_local,
                tile_idx,
                start_pos,
                head_idx,
                _batch_idx_u32,
                _kv_head_idx_u32,
                w_id,
                l_id,
                block_tile_idx,
                seq_len,
                scale_log2e,
                stagger,
                is_first,
            )

        comptime if _PERSISTENT:
            # Reference persistent grid: one block per CU; loop this CU's slice
            # of the WorkInfo stream. Each work-item = (batch, head, BM-token
            # tile, full causal KV). b1/8192 is split-free (partial_o_loc == -1,
            # direct final write); split shapes (small seq) need the reduce
            # kernel (S3) and are out of scope here.
            var cu = Int(readfirstlane(Int32(block_idx.x)))
            var w_start = Int(readfirstlane(work_indptr_ptr[cu]))
            var w_end = Int(readfirstlane(work_indptr_ptr[cu + 1]))
            for work_idx in range(w_start, w_end):
                var wbase = work_idx * 8
                var w_batch = Int(readfirstlane(work_info_ptr[wbase + 0]))
                var w_qo_start = Int(readfirstlane(work_info_ptr[wbase + 2]))
                var w_head = (
                    Int(readfirstlane(work_info_ptr[wbase + 7])) & 0xFFFF
                )
                # `is_first` (work_idx == w_start) applies the +4 prologue
                # stagger ONCE per CU; later work-items conserve the offset.
                _run_one_work(
                    w_head,
                    w_qo_start // Self.BM,
                    w_batch,
                    work_idx == w_start,
                )
                # The reference persistent back-edge is BARE — `s_addk_i32 s66,
                # 1; s_branch label_011C`, NO `s_barrier` (the reference `.co`
                # `mla_pfl_qh192_vh128_m32x8_n128x1_causal1` label_1B9F). We
                # match it: no inter-work-item barrier here. The cross-warp
                # rendezvous that gates the NEXT work-item's K/V ring DMA against
                # THIS work-item's last reads is already provided, on two
                # independent grounds:
                #   1. The next `_run_one_work` opens with `_load_q_lds_exact`,
                #      whose mandatory `vmcnt(0)+s_barrier` (the alias-scope Q
                #      fence) rendezvouses all 8 warps BEFORE the prologue
                #      `_dma_kv_split` writes ring slot 0. A fast warp cannot
                #      reach that DMA until the slow warp arrives at the Q fence,
                #      and the slow warp's arrival is downstream of its last
                #      `ds_read` of this work-item. The Q DMA itself lands in the
                #      disjoint Q@0x0 region, never the K/V ring slots, so it
                #      cannot race the prior reads.
                #   2. Independently, this work-item's last K/V `ds_read`s are
                #      RETIRED by their consuming MFMAs (the last QK/PV MFMAs
                #      take the K/V band VGPRs as operands; an MFMA stalls on
                #      `lgkmcnt` until its VGPR source is written), so the LDS
                #      slots are free before the register-only epilogue
                #      (`normalize_output` + `_store_o_to_gmem`) and the
                #      back-edge — no slot is live across the loop boundary.
                # The barrier removed here was thus redundant (it pinned a
                # workgroup rendezvous the Q fence already provides).
        else:
            # Static grid: block_idx encodes (head, BM-tile, batch).
            var block_x = Int(readfirstlane(Int32(block_idx.x)))
            var bx_div, bx_mod = divmod(block_x, _GROUP)
            var s_head_idx = bx_mod * Self.NUM_KV_HEADS + bx_div
            var s_block_tile_idx = Int(readfirstlane(Int32(block_idx.y)))
            var s_batch_idx = Int(readfirstlane(Int32(block_idx.z)))
            _run_one_work(s_head_idx, s_block_tile_idx, s_batch_idx, True)

    # ===-------------------------------------------------------------=== #
    # Ragged-batch GPU kernel entry. Mirrors MlaPrefillV2Core.ragged_kernel.
    # ===-------------------------------------------------------------=== #
    @__llvm_metadata(`llvm.amdgpu-waves-per-eu`=__mlir_attr.`"1,1"`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name(
        t"mla_prefill_v2_ragged_amd_{qkv_dtype}_{output_dtype}_BM{Self.BM}_KV{Self.KV_BLOCK}_D{Self.DEPTH}"
    )
    @staticmethod
    def ragged_kernel[
        k_nope_t: MHAOperand,
        k_rope_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        qkv_dtype: DType,
        output_dtype: DType,
    ](
        q_ptr: UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin],
        k_nope_op: k_nope_t,
        k_rope_op: k_rope_t,
        v_op: v_t,
        output_ptr: UnsafePointer[Scalar[output_dtype], MutAnyOrigin],
        mask_functor: mask_t,
        scale: Float32,
        input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
    ):
        """Ragged-batch GPU kernel entry. Per-sequence setup mirrors
        `MlaPrefillV2Core.ragged_kernel` (self-attention; `num_keys =
        start_pos + seq_len`).

        Parameters:
            k_nope_t: Compile-time operand type for the K nope
                segment.
            k_rope_t: Compile-time operand type for the K rope
                segment.
            v_t: Compile-time operand type for the V segment.
            mask_t: Compile-time mask predicate type (causal, null,
                etc.).
            qkv_dtype: Element type of the Q, K, and V tensors; must
                equal `config.dtype`.
            output_dtype: Element type of the output tensor; must
                equal `config.output_dtype`.

        Args:
            q_ptr: Pointer to the ragged-batch Q tensor data.
            k_nope_op: K (nope segment) operand; also supplies
                `cache_length` for `start_pos`.
            k_rope_op: K (rope segment) operand (unused, kept for
                contract parity).
            v_op: V operand (= nope segment of the latent cache).
            output_ptr: Pointer to the ragged-batch output tensor
                data.
            mask_functor: Per-tile mask predicate (causal / null /
                ...).
            scale: Softmax scale (`1/sqrt(d_qk)`).
            input_row_offsets_ptr: Pointer to the per-sequence row
                offset prefix-sum `[batch_size+1]` `uint32` array.
        """
        var batch_idx = block_idx.z
        var start_of_seq = Int(input_row_offsets_ptr[batch_idx])
        var end_of_seq = Int(input_row_offsets_ptr[batch_idx + 1])
        var seq_len = end_of_seq - start_of_seq

        if Int(block_idx.y) * Self.BM >= seq_len:
            return

        var start_pos = Int(k_nope_op.cache_length(batch_idx))
        var num_keys = start_pos + seq_len

        var q_batch_offset = (
            start_of_seq * Self.config.num_heads * Self.config.d_qk
        )
        var o_batch_offset = (
            start_of_seq * Self.config.num_heads * Self.config.depth
        )

        var q_ragged_layout = row_major(
            Coord(1, seq_len, Self.config.num_heads, Self.config.d_qk)
        )
        var o_ragged_layout = row_major(
            Coord(1, seq_len, Self.config.num_heads, Self.config.depth)
        )
        var q_tt = TileTensor(q_ptr + q_batch_offset, q_ragged_layout)
        var o_tt = TileTensor(output_ptr + o_batch_offset, o_ragged_layout)

        Self.run[
            k_nope_t,
            k_rope_t,
            v_t,
            mask_t,
            qkv_dtype,
            output_dtype,
            type_of(q_tt).LayoutType,
            type_of(o_tt).LayoutType,
            type_of(q_tt).Engine,
            type_of(o_tt).Engine,
            ragged=True,
        ](
            q_tt,
            k_nope_op,
            k_rope_op,
            v_op,
            o_tt,
            mask_functor,
            scale,
            Int32(num_keys),
            Int32(start_pos),
        )


@always_inline
def mla_prefill_v2_ragged[
    k_nope_t: MHAOperand,
    k_rope_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    qkv_dtype: DType,
    output_dtype: DType,
    //,
    config: MlaConfigV2,
    compile_options: StaticString = CompilationTarget[
        DeviceContext.default_device_info.target()
    ].default_compile_options(),
](
    q_ptr: UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin],
    k_nope: k_nope_t,
    k_rope: k_rope_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_dtype], MutAnyOrigin],
    mask_functor: mask_t,
    scale: Float32,
    input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
    max_prompt_len: Int,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    """Host launcher for ragged MLA prefill.

    Standard ragged-prefill signature/boilerplate (grid/block derivation,
    three operands, mask functor, scale, input row offsets).
    Grid: `(NUM_HEADS, ceildiv(max_prompt_len, BM), batch_size)`.

    Parameters:
        k_nope_t: Compile-time operand type for the K nope segment
            (inferred).
        k_rope_t: Compile-time operand type for the K rope segment
            (inferred).
        v_t: Compile-time operand type for the V segment (inferred).
        mask_t: Compile-time mask predicate type (causal, null, etc.)
            (inferred).
        qkv_dtype: Element type of the Q, K, and V tensors; must equal
            `config.dtype` (inferred).
        output_dtype: Element type of the output tensor; must equal
            `config.output_dtype` (inferred).
        config: Shape configuration (`MlaConfigV2`).
        compile_options: Compilation options for the kernel (defaults
            to the device's default compile options).

    Args:
        q_ptr: Pointer to the ragged-batch Q tensor data.
        k_nope: K nope-segment operand; supplies `cache_length` for
            `start_pos`.
        k_rope: K rope-segment operand (unused by the kernel; kept for
            contract parity).
        v: V operand (the nope segment of the latent cache).
        output_ptr: Pointer to the ragged-batch output tensor data.
        mask_functor: Per-tile mask predicate (causal / null / ...).
        scale: Softmax scale (`1/sqrt(d_qk)`).
        input_row_offsets_ptr: Pointer to the per-sequence row offset
            prefix-sum `[batch_size+1]` `uint32` array.
        max_prompt_len: Maximum prompt length in the batch; bounds the
            grid's BM-tile dimension.
        batch_size: Number of sequences in the ragged batch.
        ctx: Device context used to compile and enqueue the kernel.
    """
    comptime assert (
        qkv_dtype == config.dtype
    ), "mla_prefill_v2_ragged: `qkv_dtype` must equal `config.dtype`"
    comptime assert (
        output_dtype == config.output_dtype
    ), "mla_prefill_v2_ragged: `output_dtype` must equal `config.output_dtype`"
    comptime kernel = MlaPrefillV2[config].ragged_kernel[
        k_nope_t, k_rope_t, v_t, mask_t, qkv_dtype, output_dtype
    ]
    var compiled = ctx.compile_function[
        kernel, compile_options=compile_options
    ]()
    comptime BM = MlaPrefillV2[config].BM
    ctx.enqueue_function(
        compiled,
        q_ptr,
        k_nope,
        k_rope,
        v,
        output_ptr,
        mask_functor,
        scale,
        input_row_offsets_ptr,
        grid_dim=(
            config.num_heads,
            ceildiv(max_prompt_len, BM),
            batch_size,
        ),
        block_dim=MlaPrefillV2[config].NUM_THREADS,
    )
