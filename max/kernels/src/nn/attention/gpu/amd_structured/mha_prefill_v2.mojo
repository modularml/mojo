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
"""MhaPrefillV2: long-context BF16 MHA prefill for AMD MI355X (gfx950).

`run` interleaves QK MFMA, PV MFMA, softmax + rescale, and
K/V DMA across an explicit cluster schedule so each work class
overlaps the others' latency. 8 warps per block; each warp owns a
`Q_BLOCK_SIZE`-row stripe of Q.

## Cluster schedule

The main loop runs 8 clusters per iteration and advances `j` by 2,
so each iter processes two K/V tiles. Each cluster ends in a bare
`s_barrier`.

  | Cluster | Work                                                |
  |---------|-----------------------------------------------------|
  | C0      | QK[j-2] + tail softmax of tile (j-3)                |
  | C1      | DMA K[j] + LDS→register V[j-3] + mask (j-2)¹        |
  | C2      | PV[j-3] strip-interleaved with partial softmax(j-2) |
  | C3      | DMA V[j-1] + LDS→register K[j-1]                    |
  | C4      | QK[j-1] + tail softmax of tile (j-2)                |
  | C5      | DMA K[j+1] + LDS→register V[j-2] + mask (j-1)       |
  | C6      | PV[j-2] strip-interleaved with partial softmax(j-1) |
  | C7      | DMA V[j] + LDS→register K[j]                        |

  ¹ Non-Causal masks only. CausalMask comptime-elides the C1 mask
  call because the `max_num_tiles` cap leaves tile (j-2) naturally
  fully unmasked.

Whole-tile K is pre-loaded one iteration ahead into the persistent
`k_reg`, so the QK clusters (C0/C4) contain MFMAs + VALU only (no
in-cluster `ds_read`). The prologue primes the pipeline and runs
QK[0] + partial softmax; the 13-cluster epilogue drains the final
four tiles `N-4..N-1`, with whole-V PV (no strip split) and an
unconditional normalizer rescale before the `o / norm_vec` divide.

## Key design choices

- **Whole-tile K pre-load with consumer-side waitcnt drains.** Each
  MFMA-consumer helper opens with `s_waitcnt[lgkmcnt=0]()` so
  SIInsertWaitcnts treats the cluster as a bracket reset and the per-
  consumer `lgkmcnt` staircase collapses.

- **Kernel-scope BF16 P-cache.** Each softmax bulk-casts FP32 att to
  one persistent `att_block_bf16` register tile reused by the
  subsequent PV (avoids LLVM rematerializing the cast per use site).

- **Lazy rescale (`RESCALE_THRESHOLD=8`).** In C2/C6, when the
  running max grows by more than 8 log2 units, `o_reg *= scale_vec`
  fires between PV strip 0 and strips 1-3; strips 1-3 then
  contribute at the old scale into an already-rescaled accumulator.
  The 8 log2 cap bounds the inconsistency. When `rv_all_below`
  reports no lane exceeded the threshold, the rescale is skipped
  and `scale_vec` is reset to 1 (so the epilogue's unconditional
  multiply stays identity; see below). The epilogue's tail softmax
  applies `norm_vec *= scale_vec` *unconditionally*; the
  initialized-to-1 + reset-to-1-on-skip invariant guarantees this
  is identity unless a rescale fired in the last C2/C6.

- **Mask placement.** Tiles `0` (prologue), `(j - 2)` for each
  main-loop iter (C1, non-Causal masks only; see below), `(j - 1)`
  (C5, all masks), and `N - 3, N - 2, N - 1` (epilogue). For
  `CausalMask` the `max_num_tiles` cap guarantees odd-numbered K
  tiles in the main-loop range are naturally fully unmasked, so the
  C0/C2 path skips the mask call. Non-causal masks
  (SlidingWindow/Chunked) cannot rely on that cap, so the C1 site
  applies the mask to `att_block_1` (= QK[j-2]) before C2's partial
  softmax reads it.

- **Output transpose.** `col_l → row_l` is a zero-cost re-tag of the
  same register storage: no cross-lane permute, no data motion.

- **GQA-aware head remap.** `head_idx` is `(block_x % GROUP) *
  NUM_KV_HEADS + (block_x / GROUP)` (the transpose over the
  `(NUM_KV_HEADS, GROUP)` rectangle), so adjacent blocks visit
  different KV heads across CUs/XCDs. Bijective for any
  `NUM_HEADS == GROUP * NUM_KV_HEADS`; reduces to identity at MHA
  (`GROUP=1`) and MQA (`NUM_KV_HEADS=1`).

The cluster decomposition and overlap pattern are inspired by the
reference attention kernel.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.compile import CompilationTarget
from std.math import ceildiv
from max.gpu.sync import (
    AMDScheduleBarrierMask,
    s_waitcnt,
    schedule_group_barrier,
)
from std.memory import AddressSpace
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import (
    likely,
    llvm_intrinsic,
    readfirstlane,
)
from std.utils import StaticTuple

from layout import TensorEngine, TensorLayout, TileTensor
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
    smem_alloc,
)
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    NullMask,
)
from nn.attention.mha_operand import MHAOperand

from .buffers import _cast_f32_to_fp8_raw
from .mha_mask import MaskApplier
from .mha_softmax import OnlineSoftmax

from .mha_mma_op import MhaConfigV2, MhaMmaOp

from .iglp import (
    _iglp_opt,
    sched_barrier_exp_pairs,
    sched_barrier_pairs,
    sched_dsread_valu_pairs,
)
from std.sys import get_defined_bool, get_defined_int, size_of

# AMDGPU instruction-priority + IGroupLP scheduling helpers.


@always_inline
def _s_setprio[priority: Int16]():
    """Sets MFMA wave instruction priority (0 = normal, 1 = high)."""
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)


@always_inline
def _sched_barrier_zero():
    """`sched_barrier(0)`: hard reordering barrier that pins
    surrounding instructions to their source order."""
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@always_inline
def _asm_label[asm_str: StaticString]():
    """Emits an AMDGPU asm comment at the call site so disassembly diff
    against a reference kernel can be done by grep. Gated on
    `EMIT_ASM_LABELS`; when False this is a no-op. The
    `has_side_effect=True` + `~{memory}` form is a hard reordering
    barrier, so labels MUST be off when benchmarking; turn on for
    asm-level inspection only.

    Note on spill investigations: the inline-asm reordering barrier
    will pin spill stores/loads to their nearest label, which biases
    cluster→spill attribution toward whichever label happens to sit
    next to the high-pressure point. For diagnosing spill locations,
    keep labels OFF and instead count the natural `s_barrier`
    instructions emitted by `_s_barrier_raw` / `_cluster_barrier`;
    those are real hardware fences and survive reordering, giving an
    unbiased view of which window of source code the spill lives in.
    """

    comptime emit_asm_label = get_defined_bool["EMIT_ASM_LABELS", False]()

    comptime if emit_asm_label:
        inlined_assembly[
            asm_str,
            NoneType,
            constraints="~{memory}",
            has_side_effect=True,
        ]()


@always_inline
def _s_barrier_raw():
    """Bare `s_barrier`, with NO release/acquire fences. Mojo's stdlib
    `barrier()` would also inject `s_waitcnt vmcnt(0) lgkmcnt(0)`, which
    would defeat the partial-drain `s_waitcnt vmcnt(N) lgkmcnt(0)` we
    emit before each barrier to let DMAs cross."""
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


@always_inline
def _cluster_barrier():
    """Cluster boundary: `sched_barrier(0)` + bare `s_barrier` +
    `sched_barrier(0)`. The `sched_barrier(0)` fences pin the
    `s_barrier` to its source position so IGroupLP cannot move
    cluster-N work past the wave sync into cluster-(N+1)."""
    _sched_barrier_zero()
    _s_barrier_raw()
    _sched_barrier_zero()


struct MhaPrefillV2[config: MhaConfigV2]:
    """8-warp MHA forward kernel parameterized by `MhaConfigV2`.

    Each block runs `config.num_warps` wave64 warps that share K/V
    SMEM via cooperative DMA. Warp `w` owns Q rows `[w * q_block_size,
    (w + 1) * q_block_size)` of the block's stripe and carries its own
    register-resident attention state.

    Parameters:
        config: Shape configuration (`MhaConfigV2`).
    """

    # Config aliases — expose `config` fields as struct comptime
    # constants so body code can keep referencing `Self.Q_BLOCK_SIZE`,
    # `Self.DEPTH`, etc. unchanged.
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth
    comptime NUM_HEADS = Self.config.num_heads
    comptime NUM_KV_HEADS = Self.config.num_kv_heads
    comptime NUM_WARPS = Self.config.num_warps
    comptime RESCALE_THRESHOLD = Self.config.rescale_threshold

    comptime NUM_THREADS = Self.NUM_WARPS * 64
    comptime BM = Self.NUM_WARPS * Self.Q_BLOCK_SIZE
    comptime D_FRAG_PER_LANE = (Self.DEPTH * Self.Q_BLOCK_SIZE) // 64

    # MHA MMA operator (single source of truth for the MFMA shape, SMEM
    # sub-block geometry, register-tile layouts, and SMEM→register
    # loaders). Specialized on `config.dtype`: BF16 selects
    # `v_mfma_f32_32x32x16_bf16` with MMA_K=16; FP8 e4m3 selects
    # `v_mfma_scale_f32_32x32x64_f8f6f4` with MMA_K=64.
    comptime _MmaOp = MhaMmaOp[Self.config.dtype, Self.config]

    # Layout types per operand role — captured from `_MmaOp` so the
    # `row_major[...]` expression isn't repeated at each `RegTile`
    # signature site.
    comptime _Q_LAYOUT_T = type_of(Self._MmaOp.Q_LAYOUT)
    comptime _K_LAYOUT_T = type_of(Self._MmaOp.K_LAYOUT)
    comptime _V_LAYOUT_T = type_of(Self._MmaOp.V_LAYOUT)
    comptime _ATT_LAYOUT_T = type_of(Self._MmaOp.ATT_LAYOUT)
    comptime _ATT_BF16_SUB_LAYOUT_T = type_of(Self._MmaOp.ATT_BF16_SUB_LAYOUT)
    comptime _ATT_BF16_FULL_LAYOUT_T = type_of(Self._MmaOp.ATT_BF16_FULL_LAYOUT)
    comptime _O_LAYOUT_T = type_of(Self._MmaOp.O_LAYOUT)
    comptime _O_T_LAYOUT_T = type_of(Self._MmaOp.O_T_LAYOUT)

    # Derived block-level shape constants used across the kernel body.
    comptime _Q_ROW_STRIDE = Self.NUM_HEADS * Self.DEPTH
    comptime _KV_ROW_STRIDE = Self.NUM_KV_HEADS * Self.DEPTH
    comptime _ATT_PER_LANE = (Self.KV_BLOCK * Self.Q_BLOCK_SIZE) // 64
    """Per-lane element count for the FP32 att tile (col_l rt_32x32)."""
    comptime _ATT_HALF = Self._ATT_PER_LANE // 2
    """First-half / second-half split index for `exp2_inplace_range`."""
    comptime _PV_A_FRAG = 32 if Self.config.dtype.is_float8() else 8
    """Per-lane PV-A fragment width (= MMA_K * MMA_N / 64). Hoisted as
    a direct conditional on `config.dtype` so it resolves to a literal
    Int at MhaPrefillV2 instantiation; `Self._PV_A_FRAG`
    is the same value but the type checker doesn't fold the
    cross-struct member access at SIMD-width sites."""

    comptime _MMA_K = 64 if Self.config.dtype.is_float8() else 16
    """Per-MFMA K-dim (matches `Self._MmaOp.MMA_K`). Hoisted as a
    direct conditional so the `_NUM_PV_SUBTILES = KV_BLOCK // _MMA_K`
    division below folds to a literal Int at instantiation."""

    comptime _NUM_PV_SUBTILES = Self.KV_BLOCK // Self._MMA_K
    """Number of MMA_K-row PV strips in one K/V tile.

    BF16: 4 strips (KV_BLOCK=64 / MMA_K=16). FP8: 1 strip
    (KV_BLOCK=64 / MMA_K=64)."""

    comptime _SOFTMAX_DTYPE = (
        DType.float16 if Self.config.dtype.is_float8()
        and Self.KV_BLOCK >= 128 else DType.float32
    )
    """Softmax accumulator dtype for `att_block` register tiles.

    Phase 7.1 falsified the BF16 plan via ASM diff (commit
    `70ac6169cfc` measurements + `dump_asm=True` codegen inspection):
    **gfx950 (CDNA4) has no packed BF16 element-wise arithmetic**:
    only `V_PK_*_F32` and `V_PK_*_F16` exist. The CDNA4 ISA exposes
    BF16 ONLY via MFMA dot-products and packed conversion ops. Every
    BF16 SIMD operation lowers to per-element unpack → FP32 scalar
    op → repack. On the FP8 KV=64 BF16-softmax build:

      963 `v_cvt_pk_bf16_f32`, 401 scalar `v_max_f32_e32`,
      218 scalar `v_sub_f32_e32`, 199 scalar `v_exp_f32_e32`,
      761 `v_lshlrev_b32`, 409 `v_perm_b32`
      (vs FP32 baseline: zero scalar VOP1, 192 `v_pk_mul_f32`,
       181 `v_pk_add_f32`, 0 byte-shuffle).

    BF16 storage → 4-6x more instructions per softmax op + 388
    B/lane scratch spill → -88% TFLOPS (778 → 91) at KV=64.

    **FP16 is the right substitute**: gfx950 has the full packed
    suite (`V_PK_ADD_F16`, `V_PK_MUL_F16`, `V_PK_MAX_F16`,
    `V_PK_FMA_F16`, plus hardware-supported `v_exp_f16`). On the
    same FP8 KV=128 build, FP16 softmax shows:

      204 `v_pk_add_f16`, 141 `v_pk_max_f16`, 211 `v_exp_f16`,
      Scratch=28 B/lane (down from 656).

    Throughput:

      | Config         | FP32 softmax | FP16 softmax | Δ        |
      |----------------|--------------|--------------|----------|
      | FP8 KV=64      | 778 TFLOPS   | 712 TFLOPS   | -8.5%    |
      | FP8 KV=128     | 385 TFLOPS   | 827 TFLOPS   | +115%    |

    FP16 wins at KV=128 because the kernel is register-bound there
    (FP32 ATT_LAYOUT = 64 FP32/lane × 2 atts = 128 VGPRs alone) and
    halving the att_block storage actually frees registers; the
    conversion overhead is now packed throughput rather than
    scalar serialization. At KV=64 the kernel already fit (Scratch=0
    baseline), so the conversion overhead is pure cost.

    Therefore: FP16 softmax for FP8 + KV_BLOCK >= 128 only; FP32 for
    everything else. Precision note: FP16's 10-bit mantissa is
    comparable to BF16 (7) and exceeds FP8 (3); range is 6e-5 to
    6.5e4 which covers post-prescale QK and post-exp values for
    realistic models. Mask values that overflow saturate to
    `-FLT_MAX_F16` → exp2 → 0 (correct softmax behavior).

    All downstream code is dtype-generic on `_SOFTMAX_DTYPE` so this
    gate can be tuned per-shape without further refactoring."""

    comptime _IGLP_MFMA_BIG = get_defined_int[
        "iglp_mfma_big",
        4 if Self.config.dtype.is_float8() else 10,
    ]()
    """IGLP MFMA count for the "big" sched_barrier_pairs groups in the
    softmax-and-PV interleave (`_pv_*_with_partial_softmax`,
    `_full_softmax_unconditional`).

    Tuned to the per-kernel MFMA count: BF16 has ~192 MFMA (groups of
    10); FP8 has ~48 MFMA. FP8 (4, 2) found via sweep, +0.8% over
    the initial (3, 1) first-cut; (5, 2) ties (3, 1); (6, 3) and
    (10, 4) regress (softmax-bound, not MFMA-bound). Override via
    `-D iglp_mfma_big=N` for tuning."""

    comptime _IGLP_MFMA_SMALL = get_defined_int[
        "iglp_mfma_small",
        2 if Self.config.dtype.is_float8() else 4,
    ]()
    """IGLP MFMA count for the small post-PV-first-MFMA group in
    `_pv_strip_with_partial_softmax`.

    BF16: 4 PV MFMAs trailing the first one. FP8: 2 (raised from 1
    in the (4, 2) tuning sweep). Override via `-D iglp_mfma_small=N`
    for tuning."""

    # K SMEM swizzle: a `Swizzle(1, 0, 4)` + `Swizzle(1, 1, 4)` pair
    # in vec (16-B worker) scope composes to a byte-level `bit4 ^=
    # bit8; bit5 ^= bit9` remap — equivalent to `Swizzle(2, 4, 4)`
    # at byte scope. Both XORs use distance 4. Derived to be
    # bank-conflict-free on the reference's 32-lanes-per-row MFMA
    # access pattern (32x32x{16,64}) for `ds_read_b128` against a
    # 64-bank LDS: in every 16-lane LDS cycle, the 16 bank-quadrants
    # form a complete bijection of `{0, 4, 8, ..., 60}` for both
    # `col_offset=0` (lanes 0..31) and `col_offset=32` (lanes
    # 32..63). Verified by rocprofv3: `LDSBankConflict` drops from
    # 5.4% (the reference's distance-4+6 pair) to ~0% on FP8 KV=64.
    #
    # Earlier variants probed:
    # - The reference's original `Swizzle(1,1,4) + Swizzle(1,0,6)`
    #   (= `bit5^=bit9; bit4^=bit10`, distances 4 + 6): 5.4% conflict
    #   residual — the bit-4↔bit-10 path is not bijective at
    #   col_offset=32 (col=32 sets bit 5 of base offset and the
    #   asymmetric distance leaves a 2-way collision in the high
    #   half-warp).
    # - Ping-pong / 4-wave matmul `Swizzle(2, 5, 4)`: 10.8% conflict
    #   — pattern is tuned for that kernel's 16-lanes-per-row
    #   partition, not the 32-lanes-per-row one here.
    #
    # V uses the reference's identity-swizzle `st_8x32_s` layout.
    comptime k_swizzle = Optional(Swizzle(1, 0, 4))
    comptime k_swizzle2 = Optional(Swizzle(1, 1, 4))
    comptime v_swizzle = Optional[Swizzle](None)

    comptime KTileLoader = SubTileLoaderLDS[
        Self.config.dtype, Self.k_swizzle, Self.k_swizzle2
    ]

    # V cooperative DMA: byte-level `buffer_load_lds` over an 8-row
    # sub-tile. BK is the per-sub-tile col span in elements; keep the
    # byte size invariant across dtypes by halving the elt count for
    # BF16 (32 elts × 2 B = 64 B/row) vs FP8 (64 elts × 1 B = 64 B/row).
    comptime VTileLoader = SubTileLoaderLDS_st_8x32[
        Self.config.dtype,
        Self.KV_BLOCK,
        Self.DEPTH,
        64 if Self.config.dtype.is_float8() else 32,
        Self.NUM_THREADS,
    ]

    # Per-(batch, head) 2D layout TYPES used as `.reshape` targets after
    # slicing the input 4D / 3D tensors down to a per-(batch, head)
    # plane. Layout VALUES are constructed at the call site because the
    # `seq_len` / `num_keys` dim is runtime.
    comptime _QPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[ComptimeInt[Self._Q_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    comptime _KVPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[ComptimeInt[Self._KV_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    # Per-tile K/V layout (valid_rows x DEPTH) used by the DMA loaders.
    # `dim[0]` is a RUNTIME row count (Int32): KV_BLOCK for a full tile,
    # the partial valid-row count for the last tile when
    # `num_keys % KV_BLOCK != 0`. `make_amd_buffer_resource` reads it to
    # size the cooperative loaders' SRD `num_records`, so OOB lanes of a
    # partial last tile hardware-zero instead of reading past `num_keys`
    # into adjacent device memory (the FLUX i2i NullMask corruption). The
    # stride matches `_KV_ROW_STRIDE` (NUM_KV_HEADS * DEPTH); strides and
    # DEPTH stay comptime so only the per-tile row count costs a register.
    comptime _KvPerTileLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[ComptimeInt[Self._KV_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    # IGLP `sched_group` IDs — one per scheduling-distinct cluster. The
    # numbers must be unique within the kernel; names document which
    # cluster each ID belongs to.
    comptime _SCHED_MAIN_C0_QK_TAIL = 1
    comptime _SCHED_MAIN_C2_PV_PARTIAL = 2
    comptime _SCHED_MAIN_C4_QK_TAIL = 3
    comptime _SCHED_MAIN_C6_PV_PARTIAL = 4
    comptime _SCHED_EPI_C0_TAIL = 5
    comptime _SCHED_EPI_C2_PV_PARTIAL = 6
    comptime _SCHED_EPI_C4_TAIL = 7
    comptime _SCHED_EPI_C6_PV_PARTIAL = 8
    comptime _SCHED_EPI_C8_TAIL = 9
    comptime _SCHED_EPI_C10_FULL = 10
    comptime _SCHED_MAIN_C5_DSREAD = 11
    comptime _SCHED_EPI_C5_DSREAD = 12
    comptime _SCHED_EPI_C9_DSREAD = 13

    @staticmethod
    @always_inline
    def load_q[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[Self.config.dtype, layout, ...],
    ) -> RegTile[
        Self.config.dtype, Self._Q_LAYOUT_T, MutUntrackedOrigin
    ]:
        """Loads the warp's Q sub-tile from gmem into a row_l register
        tile via `RegTileLoader`.

        BF16 (d=128, MMA_K=16): 8 K-tiles × 1 `buffer_load_bf16x8` per
        lane per K-tile = 8 loads × 16 B each. Per-lane fragment = 8
        BF16 = 16 B fits in one buffer_load.

        FP8 (d=128, MMA_K=64): 2 K-tiles, but each base tile per lane
        is 32 FP8 = 32 B which exceeds the 16-B buffer_load_lds max.
        Splits each K-tile load into 2 × 16-elt halves (16 B each)
        targeting the first / second half of the destination cell.

        Parameters:
            layout: Layout of `q_warp_2d` (inferred).

        Args:
            q_warp_2d: Per-warp Q gmem sub-tile of shape
                `(Q_BLOCK_SIZE, DEPTH)` sliced from the per-head 2D
                Q view.
        """
        comptime _BK = Self._MmaOp.MMA_K
        comptime _num_k_tiles = Self.DEPTH // _BK
        comptime _q_thread_layout = col_major[
            Self.Q_BLOCK_SIZE, WARP_SIZE // Self.Q_BLOCK_SIZE
        ]()

        var q_reg = reg_alloc[Self.config.dtype](Self._MmaOp.Q_LAYOUT)
        var q_loader = RegTileLoader[
            Self.config.dtype, _q_thread_layout, warp_scope=True
        ](q_warp_2d)

        comptime if Self.config.dtype == .float8_e4m3fn:
            # FP8: per-lane fragment = 32 FP8 = 32 B. To match the MFMA's
            # B-operand lane layout (which is the same convention as the
            # A-operand K loader in `MhaMmaOp.load_K` FP8 32x32x64 path),
            # lane `lid` must own 32 *contiguous* K-cols in its per-lane
            # fragment: `row = lid % 32`, `col_base = (lid // 32) * 32`.
            # That is incompatible with `RegTileLoader`'s `col_major[32,
            # 2]` distribute path on a `[32, 32]` half-tile (which would
            # give lane 0 cols 0..15 in dst[0..15] AND cols 32..47 in
            # dst[16..31] — a *striped* fragment that mismatches K).
            #
            # Instead, issue 2 explicit `buffer_load` calls per lane
            # per K-tile using `AMDBufferResource` directly, with per-
            # lane base offsets matching the K loader:
            #   lo: row_offset * row_stride + col_base + 0..15
            #   hi: row_offset * row_stride + col_base + 16..31
            #
            # Per-base-tile dst storage:
            #   dst[0..15] ← lo  (cols col_base+0..15)
            #   dst[16..31] ← hi (cols col_base+16..31)
            #
            # Net per-lane fragment: row `lid % 32`, cols `(lid // 32) *
            # 32 + 0..31` (contiguous, matching K).
            comptime _row_stride = type_of(q_warp_2d).static_stride[0]
            var lid = Int(lane_id())
            var row_offset = lid % 32
            var col_base = (lid // 32) * 32

            var bc = make_amd_buffer_resource(q_warp_2d)
            var q_reg_v = q_reg.vectorize[1, 1, 16]()
            comptime for j in range(_num_k_tiles):
                # Base offset (in elements) into the Q gmem tile for
                # this lane's lo half of K-tile j.
                var base_off_lo = Int32(
                    row_offset * _row_stride + j * _BK + col_base
                )
                var base_off_hi = base_off_lo + Int32(16)
                var lo = bc.load[Self.config.dtype, 16](base_off_lo)
                var hi = bc.load[Self.config.dtype, 16](base_off_hi)
                # dst[0..15] = lo, dst[16..31] = hi via two halves of
                # the per-base-tile vectorize-16 register.
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

    # Comptime gate: prescale Q by `scale * log2e` at load time so the
    # downstream `exp2(QK^T)` is the correct softmax. Disabled for FP8
    # because the FP32 → FP8 quantization of the prescaled Q discards
    # too much precision (see `amd_structured/mha_prefill.mojo:95-101`);
    # for FP8 the scale lands post-QK on the att tile.
    comptime prescale_q = not Self.config.dtype.is_float8()

    @staticmethod
    @always_inline
    def _load_q_and_scale[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[Self.config.dtype, layout, ...],
        scale_log2e: Float32,
    ) -> RegTile[Self.config.dtype, Self._Q_LAYOUT_T, MutUntrackedOrigin]:
        """Loads Q from gmem and (when `Self.prescale_q` is True) prescales
        it by `scale * log2e`.

        The multiply is done per-fragment in FP32 then cast back to the
        input dtype, so only one FP32 fragment is alive at a time. The
        downstream QK MFMA consumes `q_reg` as B in pre-transpose form
        via `mma[swap_b=True]`; no explicit transpose tile needed.

        When `Self.prescale_q` is False (FP8 path), `scale_log2e` is
        unused here; the scale lands on the att tile post-QK."""
        var q_reg = Self.load_q(q_warp_2d)

        comptime if Self.prescale_q:
            comptime _H = Self._Q_LAYOUT_T.static_shape[0]
            comptime _W = Self._Q_LAYOUT_T.static_shape[1]
            comptime _F = Self._Q_LAYOUT_T.static_shape[2]
            var q_v = q_reg.vectorize[1, 1, _F]()
            comptime assert q_v.flat_rank == 3
            comptime for h in range(_H):
                comptime for w in range(_W):
                    q_v[h, w, 0] = (
                        q_v[h, w, 0].cast[.float32]() * scale_log2e
                    ).cast[Self.config.dtype]()

        return q_reg

    # K/V DMA helpers: issue the gmem→LDS load only, with no barrier or
    # waitcnt so the caller can overlap MFMAs with in-flight DMAs. The
    # per-tile gmem_tile is supplied by the caller (built via
    # `_make_k_tile` / `_make_v_tile` from the MHAOperand), so the helpers
    # are agnostic to paged-vs-contiguous KV.

    @staticmethod
    @always_inline
    def _make_k_tile[
        k_t: MHAOperand,
        //,
    ](
        k_op: k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        num_keys: Int,
    ) -> TileTensor[Self.config.dtype, Self._KvPerTileLayoutT, ImmutAnyOrigin]:
        """Builds the per-tile K gmem TileTensor at `(batch, t*KV_BLOCK,
        kv_head, 0)` via `MHAOperand.block_paged_tile`. For
        LayoutTensorMHAOperand this resolves to a pointer-arithmetic
        offset into a contiguous buffer; for KVCacheMHAOperand it
        resolves through the page table."""
        comptime assert (
            k_t.dtype == Self.config.dtype
        ), "MhaPrefillV2: K dtype must equal `config.dtype`"
        # `num_keys` clamps the runtime `dim[0]` to the valid K extent:
        # KV_BLOCK for a full tile, the partial count for the last tile,
        # <= 0 for a phantom tile entirely past `num_keys` (the SRD
        # collapses to a zero-byte resource → all lanes hardware-zero).
        var valid_rows = min(Self.KV_BLOCK, num_keys - t * Self.KV_BLOCK)
        # rebind: comptime k_t.dtype == config.dtype (asserted above), so
        # the returned TileTensor's dtype parameter is statically
        # config.dtype.
        return rebind[
            TileTensor[
                Self.config.dtype, Self._KvPerTileLayoutT, ImmutAnyOrigin
            ]
        ](
            k_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._KvPerTileLayoutT(
                    Coord(Int32(valid_rows), Idx[Self.DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE], Idx[1]),
                ),
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
        num_keys: Int,
    ) -> TileTensor[Self.config.dtype, Self._KvPerTileLayoutT, ImmutAnyOrigin]:
        """Builds the per-tile V gmem TileTensor (see `_make_k_tile`).

        `num_keys` clamps the runtime `dim[0]` identically to
        `_make_k_tile` so V's partial-last-tile OOB rows hardware-zero
        rather than leaking stale V into PV (the FLUX i2i bug)."""
        comptime assert (
            v_t.dtype == Self.config.dtype
        ), "MhaPrefillV2: V dtype must equal `config.dtype`"
        var valid_rows = min(Self.KV_BLOCK, num_keys - t * Self.KV_BLOCK)
        return rebind[
            TileTensor[
                Self.config.dtype, Self._KvPerTileLayoutT, ImmutAnyOrigin
            ]
        ](
            v_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._KvPerTileLayoutT(
                    Coord(Int32(valid_rows), Idx[Self.DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE], Idx[1]),
                ),
            )
        )

    @staticmethod
    @always_inline
    def _dma_k[
        k_t: MHAOperand,
        //,
    ](
        k_smem_slot: SMemTile[Self.config.dtype, ...],
        k_op: k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        num_keys: Int,
        w_id: Int,
        l_id: Int,
    ):
        """Issues the K[t] DMA into `k_smem_slot` from the MHAOperand.

        Partition: at d=128 the K tile holds 8 `K_SUB_ROWS x K_SUB_COLS`
        sub-blocks (2 row-tiles × 4 col-tiles), matching `NUM_WARPS=8`
        → 1 warp / 1 sub-block; each warp loads a full 32×32 sub-block
        via two 16-row internal strips. At d=64 the K tile holds only
        4 sub-blocks (2 × 2) → 2 warps cooperate per sub-block, each
        loading a 16-row half-strip. The per-warp `SubTileLoaderLDS`
        bakes the reference's two-XOR `st_32x32_s` swizzle into the
        DRAM-source lane mapping; `_MmaOp.load_K` unswizzles on read.

        Loader construction is per-tile: the buffer resource is bound
        to this tile's pointer + bounds, so the same helper composes
        with paged KV (where adjacent tiles live in different DRAM
        pages) without a separate code path. Cost is ~4 SGPR ops per
        call to materialize the SRD; negligible vs the DMA latency."""
        comptime _K_SUB_ROWS = Self._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._MmaOp.K_SUB_COLS
        comptime _num_kv_subblocks = (
            (Self.KV_BLOCK // _K_SUB_ROWS) * (Self.DEPTH // _K_SUB_COLS)
        )
        comptime _warps_per_subblock = Self.NUM_WARPS // _num_kv_subblocks
        comptime _rows_per_warp = _K_SUB_ROWS // _warps_per_subblock
        comptime _num_block_cols_k = Self.DEPTH // _K_SUB_COLS
        comptime assert (
            Self.NUM_WARPS == _num_kv_subblocks * _warps_per_subblock
        ), (
            "MhaPrefillV2 K DMA: NUM_WARPS must divide evenly into the"
            " K sub-block grid"
        )

        var subblock_id, row_strip = divmod(w_id, _warps_per_subblock)
        var sub_row, sub_col = divmod(subblock_id, _num_block_cols_k)

        var k_gmem_tile = Self._make_k_tile(
            k_op, batch_idx, kv_head_idx, t, num_keys
        )
        var k_loader = Self.KTileLoader(k_gmem_tile)
        var k_src = k_gmem_tile.tile[_K_SUB_ROWS, _K_SUB_COLS](
            sub_row, sub_col
        ).tile[_rows_per_warp, _K_SUB_COLS](row_strip, 0)
        # Inline pass-through: byte-identical to the previous internal
        # `Int(src_partitions.ptr) - dram_base` computation. No hoist
        # needed — this loader serves a single K DMA per tile.
        k_loader.load(
            k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](subblock_id, 0).tile[
                _rows_per_warp, _K_SUB_COLS
            ](row_strip, 0),
            k_src,
            scalar_offset=Int(k_src.ptr) - k_loader.bc.get_base_ptr(),
        )

    @staticmethod
    @always_inline
    def _dma_v[
        v_t: MHAOperand,
        //,
    ](
        v_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
        v_op: v_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        num_keys: Int,
        w_id: Int,
        l_id: Int,
    ):
        """Issues the V[t] DMA into `v_smem_slot` from the MHAOperand.
        V uses identity swizzle and the cooperative loader handles
        thread-id mapping over the whole `KV_BLOCK x DEPTH` tile.

        Loader construction is per-tile (see `_dma_k` docstring)."""
        var v_gmem_tile = Self._make_v_tile(
            v_op, batch_idx, kv_head_idx, t, num_keys
        )
        var v_loader = Self.VTileLoader(v_gmem_tile)
        # Inline pass-through: loader is built from `v_gmem_tile` and
        # passes the SAME tile as src, so `scalar_offset` evaluates to
        # 0 — byte-identical to the previous internal computation.
        v_loader.load(
            v_smem_slot,
            v_gmem_tile,
            w_id,
            l_id,
            scalar_offset=Int(v_gmem_tile.ptr) - v_loader.bc.get_base_ptr(),
        )

    # Whole-tile K pre-load + consumer-side waitcnt drain. `_load_k_reg`
    # runs in a dedicated cluster; `_qk_with_kreg` consumes the result
    # without any in-cluster `ds_read`.

    @staticmethod
    @always_inline
    def _load_k_reg(
        mut k_reg: RegTile[
            Self.config.dtype, Self._K_LAYOUT_T, MutUntrackedOrigin
        ],
        k_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
    ):
        Self._MmaOp.load_K(k_reg, k_smem_slot)

    @staticmethod
    @always_inline
    def _qk_with_kreg(
        mut att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        mut k_reg: RegTile[
            Self.config.dtype, Self._K_LAYOUT_T, MutUntrackedOrigin
        ],
        mut q_reg: RegTile[
            Self.config.dtype, Self._Q_LAYOUT_T, MutUntrackedOrigin
        ],
        scale_log2e: Float32,
    ):
        """QK MFMA over pre-loaded `k_reg`. The opening
        `s_waitcnt[lgkmcnt=0]()` collapses the per-VGPR staircase that
        SIInsertWaitcnts would otherwise emit at each consumer.

        BF16 path (`_SOFTMAX_DTYPE == FP32`): `att_block` IS the FP32
        MFMA target. Q was prescaled by `scale_log2e` at load time
        (`_load_q_and_scale`), so `scale_log2e` here is unused.

        FP8 path (`_SOFTMAX_DTYPE == BF16`, A/B re-probe):
        `Self.prescale_q == False` (FP8 prescale skipped to avoid the
        FP32 → FP8 quantization precision loss, sub-step 5). MFMA
        writes a short-lived FP32 temp; `scale_log2e` multiplies in
        FP32 then casts to BF16 into the persistent `att_block`.
        Fusing the multiply + cast into one per-fragment expression
        keeps the FP32 lifetime per-iter so the BF16 final value can
        displace it (`att_block` is the persistent caller-owned
        destination, which keeps LLVM from rematerializing the cast)."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        comptime _ATT_H = Self._ATT_LAYOUT_T.static_shape[0]
        comptime _ATT_W = Self._ATT_LAYOUT_T.static_shape[1]
        comptime if Self._SOFTMAX_DTYPE == .float32:
            _ = att_block.fill(0)
            Self._MmaOp.mma_QK(att_block, k_reg, q_reg)
            comptime if not Self.prescale_q:
                # BF16 attention path: att_block is FP32. Apply
                # post-QK scale in place. The explicit
                # `.cast[Self._SOFTMAX_DTYPE]()` on the scalar +
                # `att_v.ElementType(...)` broadcast keeps the
                # expression type-checking on the FP8 path as well,
                # where this branch is comptime-dead but Mojo still
                # type-checks both arms.
                var att_v = att_block.vectorize[1, 1, 16]()
                var scale = att_v.ElementType(
                    scale_log2e.cast[Self._SOFTMAX_DTYPE]()
                )
                comptime for h in range(_ATT_H):
                    comptime for w in range(_ATT_W):
                        att_v[h, w, 0] = att_v[h, w, 0] * scale
        else:
            # FP8 attention path: MFMA + scale in FP32, cast to BF16
            # in a fused per-fragment expression. Always applies the
            # post-QK scale (the FP8 path has `prescale_q == False`).
            var att_fp32 = reg_alloc[.float32](Self._MmaOp.ATT_LAYOUT)
            _ = att_fp32.fill(0)
            Self._MmaOp.mma_QK(att_fp32, k_reg, q_reg)
            var att_fp32_v = att_fp32.vectorize[1, 1, 16]()
            var att_bf16_v = att_block.vectorize[1, 1, 16]()
            comptime for h in range(_ATT_H):
                comptime for w in range(_ATT_W):
                    att_bf16_v[h, w, 0] = (
                        att_fp32_v[h, w, 0] * scale_log2e
                    ).cast[Self._SOFTMAX_DTYPE]()

    @staticmethod
    @always_inline
    def _load_v_reg(
        mut v_reg: RegTile[
            Self.config.dtype, Self._V_LAYOUT_T, MutUntrackedOrigin
        ],
        v_smem_slot: SMemTile[Self.config.dtype, _, MutUntrackedOrigin, ...],
    ):
        """LDS→register V load. No waitcnt: the consumer PV cluster
        drains LDS via its own `s_waitcnt[lgkmcnt=0]()`."""
        Self._MmaOp.load_V(v_reg, v_smem_slot)

    @staticmethod
    @always_inline
    def _att_bf16_subtile_jit[
        subtile_idx: Int,
    ](
        att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
    ) -> RegTile[
        Self.config.dtype, Self._ATT_BF16_SUB_LAYOUT_T, MutUntrackedOrigin
    ]:
        """JIT-cast one PV-A subtile (MMA_K-row strip) from `att_block`.
        The narrow lifetime lets RA fold the cast registers into the
        surrounding PV MFMA.

        BF16 attention path (MMA_K=16, src is FP32): one source strip
        (16 FP32/lane) casts to 16 BF16/lane, then a half-slice (8
        BF16/lane) feeds one PV-A sub-tile. Each src strip contributes
        to 2 sub-tiles via `(_strip, _half) = divmod(subtile_idx, 2)`.

        FP8 attention path (MMA_K=64, src is BF16 in sub-step 8): two
        source strips (32 BF16/lane total = 16 VGPR/lane, vs 32 VGPR
        when src was FP32) cast through FP32 to 32 FP8/lane and JOIN
        into one PV-A sub-tile. `subtile_idx` is always 0 since
        `_NUM_PV_SUBTILES=1`. gfx950 has no direct BF16→FP8 v_cvt;
        the BF16 → FP32 → FP8 round-trip emits `v_cvt_f32_bf16` +
        `v_cvt_pk_fp8_f32` per fragment."""
        var result = reg_alloc[Self.config.dtype](
            Self._MmaOp.ATT_BF16_SUB_LAYOUT
        )
        var src_v = att_block.vectorize[1, 1, 16]()
        comptime if Self.config.dtype == .float8_e4m3fn:
            # FP8: 2 source strips JOIN into one 32-FP8/lane sub-tile.
            # src is BF16 (sub-step 8): BF16 → FP32 → FP8 round-trip.
            # Bare `v_cvt_pk_fp8_f32` for FP32→FP8 — P ∈ [0, 1] post-
            # softmax is provably bounded + non-NaN, so the generic
            # `SIMD.cast`'s v_med3_f32 / v_cmp_u_f32 / v_cndmask
            # saturate+NaN-scrub wrapper is unnecessary and would
            # force the ±448 saturation constants into VGPRs across
            # cluster boundaries.
            var dst_v = result.vectorize[1, 1, 32]()
            var fp32_lo = src_v[0, 0, 0].cast[.float32]()
            var fp32_hi = src_v[1, 0, 0].cast[.float32]()
            var fp8_lo = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_lo)
            var fp8_hi = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_hi)
            dst_v[0, 0, 0] = fp8_lo.join(fp8_hi)
        else:
            # BF16 attention path: src is FP32, dst is BF16. One strip
            # (16 FP32/lane) casts to 16 BF16 then a half-slice (8
            # BF16/lane) feeds one PV-A sub-tile. 4 sub-tiles total
            # via `(_strip, _half) = divmod(idx, 2)`.
            var dst_v = result.vectorize[1, 1, 8]()
            comptime _strip, _half = divmod(subtile_idx, 2)
            var bf16 = src_v[_strip, 0, 0].cast[Self.config.dtype]()
            dst_v[0, 0, 0] = bf16.slice[8, offset=_half * 8]()
        return result

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
        """Bulk-casts `att_block` to the PV-A input dtype, writing into
        the caller-provided persistent destination `dst`.

        Must be called in the QK+softmax cluster, NOT inside the consumer
        PV cluster: placing the cast before the `s_barrier` lets the
        `v_cvt`s overlap with the barrier and the next cluster's DMA so
        PV's MFMAs run back-to-back. The persistent destination
        prevents LLVM from rematerializing the cast at each PV use site.

        BF16 attention path (src is FP32): each of the 2 source strips
        (16 FP32/lane) feeds 2 sub-tiles via a half-slice. 4 sub-tiles
        total for KV_BLOCK=64.

        FP8 attention path (src is BF16 in sub-step 8): the 2 source
        strips (32 BF16/lane total) cast through FP32 and JOIN into 1
        sub-tile (32 FP8/lane): `_NUM_PV_SUBTILES=1`. BF16 → FP32 →
        FP8 because gfx950 has no direct BF16→FP8 v_cvt."""
        var src_v = att_block.vectorize[1, 1, 16]()
        comptime if Self.config.dtype == .float8_e4m3fn:
            # FP8 path: src is BF16; cast through FP32 → FP8 and JOIN.
            # Bare `v_cvt_pk_fp8_f32` (see `_att_bf16_sub` for the
            # safety/perf rationale).
            var dst_v = dst.vectorize[1, 1, 32]()
            var fp32_lo = src_v[0, 0, 0].cast[.float32]()
            var fp32_hi = src_v[1, 0, 0].cast[.float32]()
            var fp8_lo = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_lo)
            var fp8_hi = _cast_f32_to_fp8_raw[Self.config.dtype](fp32_hi)
            dst_v[0, 0, 0] = fp8_lo.join(fp8_hi)
            # Previously held a `~{memory}` clobber to defeat LLVM cast
            # rematerialization. With the bare-cvt simplification the
            # cast is a single `v_cvt_pk_fp8_f32` per fragment and the
            # ±448 saturation constants no longer occupy VGPRs, so even
            # if LLVM rematerializes the cost is bounded — the clobber
            # is no longer load-bearing.
        else:
            # BF16 attention path: src is FP32, dst is BF16. Each of the
            # 2 source strips (16 FP32/lane) casts to 16 BF16 then
            # half-slices into 2 sub-tiles. 4 sub-tiles total for
            # `_NUM_PV_SUBTILES = KV_BLOCK // MMA_K = 4`.
            # The per-subtile memory clobber for BF16 cast remat is
            # also removed alongside the FP8 one; BF16 cast is FP32 →
            # truncate-mantissa (no clamp/NaN wrapper) so remat cost is
            # trivially bounded if it happens at all.
            var dst_v = dst.vectorize[1, 1, 8]()
            comptime for sub in range(Self._NUM_PV_SUBTILES):
                comptime _strip, _half = divmod(sub, 2)
                var bf16 = src_v[_strip, 0, 0].cast[Self.config.dtype]()
                dst_v[sub, 0, 0] = bf16.slice[8, offset=_half * 8]()

    @staticmethod
    @always_inline
    def _pv_whole(
        v_reg: RegTile[Self.config.dtype, Self._V_LAYOUT_T, MutUntrackedOrigin],
        att_bf16_full: RegTile[
            Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
        ],
        mut o_reg: RegTile[.float32, Self._O_LAYOUT_T, MutUntrackedOrigin],
    ):
        """Whole-V PV MFMA over a pre-cast `att_bf16_full`. No fused
        softmax; used by the epilogue PV clusters."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        comptime for i in range(Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, Self._PV_A_FRAG](
                i, 0, 0
            )
            var att_sub = att_bf16_full.tile[1, 1, Self._PV_A_FRAG](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)

    @staticmethod
    @always_inline
    def _pv_strip_with_partial_softmax[
        sched_group: Int,
    ](
        v_reg: RegTile[Self.config.dtype, Self._V_LAYOUT_T, MutUntrackedOrigin],
        mut att_bf16_full: RegTile[
            Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
        ],
        mut o_reg: RegTile[.float32, Self._O_LAYOUT_T, MutUntrackedOrigin],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
        mut att_block_qk: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
    ) -> Bool:
        """Main-loop C2/C6 body: strip-interleaved PV (using pre-loaded
        `v_reg`) + partial softmax of the next tile. Returns whether a
        rescale of `norm_vec` is pending for the next QK tail.

        RESCALE CONSISTENCY (#87284): when the rescale fires,
        `lazy_rescale_decision` rescales BOTH `o_reg` AND `att_bf16_full`
        by `scale_vec` between PV strip 0 and strips 1..3. Strips 1..3
        then consume `att_bf16_full` at the post-rescale scale,
        consistent with the rescaled `o_reg`. Without the `att_bf16_full`
        rescale, strips 1..3 over-contribute at the OLD scale: a bounded
        artifact that corrupts wide-dynamic-range attention (FLUX
        NullMask no-QK-norm prefill). Skipped on the `_rv_all_below` path.

        Softmax recurrence steps owned by `softmax`:
        `col_max_acc` → `lazy_rescale_decision` → `sub_max`. See
        `OnlineSoftmax.lazy_rescale_decision` for the SCALE_VEC
        INVARIANT that protects the epilogue's unconditional
        `norm_vec *= scale_vec` from `Inf` on non-Causal masks.
        """
        s_waitcnt[lgkmcnt=UInt32(0)]()
        _s_setprio[Int16(1)]()

        # PV strip 0.
        var v_sub_0 = v_reg.tile[1, Self.DEPTH // 32, Self._PV_A_FRAG](0, 0, 0)
        var att_sub_0 = att_bf16_full.tile[1, 1, Self._PV_A_FRAG](0, 0, 0)
        Self._MmaOp.mma_PV(o_reg, v_sub_0, att_sub_0)

        # col_max + lazy rescale decision.
        softmax.col_max_acc(att_block_qk)
        # IGLP: 4×(1 MFMA + 5 VALU) interleaves the next 4 PV MFMAs with
        # the col_max + rescale VALU work.
        sched_barrier_pairs[
            Self._IGLP_MFMA_SMALL, valu_cnt=5, group=sched_group
        ]()
        var pending_scale = softmax.lazy_rescale_decision(
            o_reg, att_bf16_full, Self.RESCALE_THRESHOLD
        )

        # PV strips 1..3.
        comptime for i in range(1, Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, Self._PV_A_FRAG](
                i, 0, 0
            )
            var att_sub = att_bf16_full.tile[1, 1, Self._PV_A_FRAG](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)

        # `att - max_vec` + first-half exp2 in preparation for the QK
        # tail softmax cluster that consumes `att_block_qk` next.
        softmax.sub_max(att_block_qk)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_qk)

        # IGLP: trailing PV MFMAs interleaved with `sub_col` (VALU) and
        # first-half `exp2` (TRANS).
        sched_barrier_pairs[
            Self._NUM_PV_SUBTILES + 2, valu_cnt=5, group=sched_group
        ]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        _s_setprio[Int16(0)]()
        return pending_scale

    @staticmethod
    @always_inline
    def _pv_whole_with_partial_softmax[
        sched_group: Int,
    ](
        v_reg: RegTile[Self.config.dtype, Self._V_LAYOUT_T, MutUntrackedOrigin],
        att_bf16_full: RegTile[
            Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
        ],
        mut o_reg: RegTile[.float32, Self._O_LAYOUT_T, MutUntrackedOrigin],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
        mut att_block_qk: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
    ):
        """Epilogue C2/C6 body: whole-V PV then UNCONDITIONAL rescale +
        partial softmax. Unlike `_pv_strip_with_partial_softmax`, the
        rescale fires AFTER all PV MFMAs so there is no strip-vs-rescale
        inconsistency; all of V's contribution lands at the old scale
        before `o_reg` is rescaled to the new one."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        _s_setprio[Int16(1)]()
        comptime for i in range(Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, Self._PV_A_FRAG](
                i, 0, 0
            )
            var att_sub = att_bf16_full.tile[1, 1, Self._PV_A_FRAG](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)
        softmax.col_max_acc(att_block_qk)
        softmax.update_scale_unconditional()
        softmax.sub_max(att_block_qk)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_qk)
        # IGLP: interleave PV MFMAs with sub_col + mul_col (VALU) and
        # first-half exp2 (TRANS).
        sched_barrier_pairs[
            Self._IGLP_MFMA_BIG, valu_cnt=5, group=sched_group
        ]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        softmax.rescale_output(o_reg)
        _s_setprio[Int16(0)]()

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
        """Writes the FP32 row_l rt_32x32 accumulator to gmem via
        `RegTileEpilogue`, casting per-lane to `output_dtype`.

        The per-lane → (q_in_tile, output_col) mapping follows the
        rt_32x32 row_l fragment topology: lanes `[0, 32)` and `[32, 64)`
        own the two halves of the depth, offset by 4 to interleave.
        Since the col_l → row_l transpose is a zero-cost re-tag, the
        FP32-path stored values are bit-identical to the col_l
        accumulator; the BF16-path cast is a per-lane `v_cvt_pkrtz`
        (or scalar `v_cvt`) emitted just before the gmem store.

        `valid_q_rows_in_warp` is the M-bound for this warp's slice of
        the BM tile. For a full tile pass `Q_BLOCK_SIZE`; for a partial
        last tile of a sequence whose length isn't a multiple of `BM`,
        callers pass `clamp(seq_len - block_tile_idx * BM - warp_id *
        Q_BLOCK_SIZE, 0, Q_BLOCK_SIZE)`. Stores at
        `q_in_tile >= valid_q_rows_in_warp` are skipped; RegTileEpilogue
        leaves the M check to the caller (line 1832-1835), and the
        per-row store gate here is what makes the writer correct for
        partial-Q tiles."""
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

    @staticmethod
    @always_inline
    def _tail_softmax_unconditional[
        sched_group: Int,
    ](
        mut att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
    ):
        """Epilogue tail softmax: second-half `exp2` + UNCONDITIONAL
        `norm_vec *= scale_vec` + `col_sum_acc`. No BF16 cast; the
        consumer PV JIT-casts `att_block` per subtile inline.

        The unconditional `norm_vec *= scale_vec` relies on the
        invariant that `scale_vec` is 1 whenever no rescale fired in
        the most recent C2/C6 (maintained by
        `OnlineSoftmax.lazy_rescale_decision`'s skip branch).
        """
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block
        )
        softmax.apply_unconditional_norm_rescale()
        softmax.col_sum_acc(att_block)
        # IGLP: interleave QK MFMAs with second-half exp2 (TRANS) and
        # col_sum_acc (VALU).
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        sched_barrier_pairs[
            Self._IGLP_MFMA_BIG, valu_cnt=5, group=sched_group
        ]()

    @staticmethod
    @always_inline
    def _qk_tail_softmax_cluster[
        sched_group: Int,
    ](
        mut att_block_qk: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        mut att_block_softmax: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        mut att_block_bf16: RegTile[
            Self.config.dtype, Self._ATT_BF16_FULL_LAYOUT_T, MutUntrackedOrigin
        ],
        mut k_reg: RegTile[
            Self.config.dtype, Self._K_LAYOUT_T, MutUntrackedOrigin
        ],
        mut q_reg: RegTile[
            Self.config.dtype, Self._Q_LAYOUT_T, MutUntrackedOrigin
        ],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
        pending_scale: Bool,
        scale_log2e: Float32,
    ):
        """Main-loop C0/C4 body: QK MFMA (consuming pre-loaded `k_reg`)
        + tail softmax of the previous tile + bulk BF16 cast into the
        persistent `att_block_bf16` for the next PV cluster.

        The `exp2` → optional rescale → `col_sum_acc` → cast order and
        the `norm_pre` register stash (internal to
        `apply_norm_rescale_if_pending` / `col_sum_acc`) are tuned so
        the cast `v_cvt`s sit at the very end of the cluster and
        overlap the next barrier + DMA.

        `scale_log2e` is unused for BF16 (Q was prescaled at load time)
        and applied post-QK on `att_block_qk` for FP8."""
        Self._qk_with_kreg(att_block_qk, k_reg, q_reg, scale_log2e)
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block_softmax
        )
        softmax.apply_norm_rescale_if_pending(pending_scale)
        softmax.col_sum_acc(att_block_softmax)
        Self._att_bf16_full(att_block_bf16, att_block_softmax)
        # IGLP: interleave QK MFMAs with second-half exp2 (TRANS) and
        # col_sum_acc + bulk BF16 cast `v_cvt`s (VALU).
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        sched_barrier_pairs[
            Self._IGLP_MFMA_BIG, valu_cnt=5, group=sched_group
        ]()

    @staticmethod
    @always_inline
    def _full_softmax_unconditional[
        sched_group: Int,
    ](
        mut att_block: RegTile[
            Self._SOFTMAX_DTYPE, Self._ATT_LAYOUT_T, MutUntrackedOrigin
        ],
        mut softmax: OnlineSoftmax[Self._SOFTMAX_DTYPE],
    ):
        """Epilogue full softmax: both halves of `exp2` + UNCONDITIONAL
        norm rescale + `col_sum`. No cast; the consumer `_pv_whole`
        reuses an already-staged `att_block_bf16`."""
        softmax.col_max_acc(att_block)
        softmax.update_scale_unconditional()
        softmax.sub_max(att_block)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block)
        # IGLP: interleave PV-whole MFMAs with sub_col + mul_col (VALU)
        # and first-half exp2 (TRANS).
        sched_barrier_pairs[
            Self._IGLP_MFMA_BIG, valu_cnt=5, group=sched_group
        ]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block
        )
        softmax.apply_unconditional_norm_rescale()
        softmax.col_sum_acc(att_block)

    # `amdgpu-waves-per-eu` accepts a `"MIN,MAX"` string. The
    # `llvm.<name>` namespace in `@__llvm_metadata` routes through
    # `LowerKGENToLLVM.cpp`'s LLVM-passthrough path (drop `llvm.`
    # prefix, build an `(attr_name, attr_value)` ArrayAttr entry on
    # `llvm.func`'s `passthrough` slot). Lands on the LLVM function
    # without any MLIR ROCDL-dialect or LLVM-tarball patches.
    @__llvm_metadata(`llvm.amdgpu-waves-per-eu`=__mlir_attr.`"2,2"`)
    # The Modular-local LLVM patch at
    # `bazel/third-party/llvm-agpr-alloc-respect-author.patch` exposes a
    # `rocdl.agpr_alloc_min_required` decorator opting a function into
    # AGPR-form MFMA codegen. We intentionally omit it here: gfx950 has
    # no AGPR-side ALU, so online softmax's `o_reg *= exp2(...)` round-
    # trips every accumulator value through VGPRs (~25x regression
    # at MQA seq=8192 with MIN==MAX==64).
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name(
        t"mha_prefill_v2_amd_{q_dtype}_{output_dtype}_BM{Self.BM}_KV{Self.KV_BLOCK}_D{Self.DEPTH}"
    )
    @staticmethod
    def run[
        k_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        q_dtype: DType,
        output_dtype: DType,
        q_layout: TensorLayout,
        o_layout: TensorLayout,
        q_engine: TensorEngine,
        o_engine: TensorEngine,
        ragged: Bool = False,
        sink: Bool = False,
    ](
        q: TileTensor[q_dtype, q_layout, ImmutAnyOrigin, Engine=q_engine],
        k: k_t,
        v: v_t,
        o: TileTensor[output_dtype, o_layout, MutAnyOrigin, Engine=o_engine],
        mask_functor: mask_t,
        scale: Float32,
        num_keys: Int32,
        start_pos: Int32,
        sink_weights_ptr: UnsafePointer[Scalar[q_dtype], ImmutAnyOrigin],
    ):
        """Multi-block 8-warp MHA forward (inference-only).

        Grid: `(NUM_HEADS, ceildiv(seq_len, BM), batch)`. Each block
        owns one `(batch, head, BM-tile)` slice; the 8 warps within
        split the BM-tile's Q rows.

        Expected layouts / shapes:

        - `q`, `o`: `(batch, seq_len, NUM_HEADS, DEPTH)` row-major
          TileTensor. `o`'s dtype matches `config.output_dtype`: BF16
          for the production dispatcher (which holds a BF16 output
          buffer) or FP32 if the caller wants the unnormalized
          accumulator.
        - `k`, `v`: any `MHAOperand` whose `block_paged_tile[KV_BLOCK]`
          returns `(KV_BLOCK, DEPTH)` tiles per `(batch, t*KV_BLOCK,
          kv_head, 0)`. `LayoutTensorMHAOperand` for contiguous test /
          bench buffers; `KVCacheMHAOperand` for paged production
          caches (`page_size >= KV_BLOCK = 64`).

        `batch` and `seq_len` / `num_keys` may be dynamic;
        `NUM_HEADS`, `NUM_KV_HEADS`, `DEPTH` must be static.
        `NUM_HEADS` must be a multiple of `NUM_KV_HEADS`
        (GROUP = `NUM_HEADS // NUM_KV_HEADS`).

        Parameters:
            k_t: K operand type (inferred); any `MHAOperand` whose
                `block_paged_tile` returns `(KV_BLOCK, DEPTH)` tiles.
            v_t: V operand type (inferred); same tile contract as
                `k_t`.
            mask_t: Mask functor type (inferred); selects the
                per-tile masking strategy (causal, sliding-window,
                null, etc.).
            q_dtype: Element dtype of `q` (inferred); must equal
                `config.dtype`.
            output_dtype: Element dtype of `o` (inferred); must
                equal `config.output_dtype`.
            q_layout: Layout of the `q` TileTensor (inferred).
            o_layout: Layout of the `o` TileTensor (inferred).
            q_engine: Engine of the `q` TileTensor (inferred).
            o_engine: Engine of the `o` TileTensor (inferred).
            ragged: Whether `q` is a per-sequence slice in a packed
                ragged batch (defaults to False).
            sink: Whether to seed the online softmax with
                attention-sink weights (defaults to False).

        Args:
            q: Q tile tensor.
            k: K operand (MHAOperand).
            v: V operand (MHAOperand).
            o: Output tile tensor (`config.output_dtype`, same shape
                as `q`).
            mask_functor: Per-tile mask predicate (causal, sliding-window,
                etc.). Evaluated inside the QK→softmax cluster; identity
                for unmasked attention.
            scale: Softmax scale (`1/sqrt(DEPTH)`).
            num_keys: Runtime length of the K/V sequence.
            start_pos: Position of the first Q row in the global sequence:
                non-zero for prefill chunks of a longer generation.
                Used by the mask functor to compute the causal cutoff.
            sink_weights_ptr: Per-q-head attention-sink scalar weights.
                Read only when the comptime `sink` parameter is True;
                the non-sink path comptime-elides the load, so callers may
                pass `Pointer[...].unsafe_dangling()` when
                `sink=False`. Indexed by `head_idx` once per block at
                init time, cast to FP32, multiplied by `log2e` to land
                in the kernel's log2-units rowmax, and seeded into
                `max_vec` / `max_vec_prev` / `norm_vec` so the hot loop
                stays sink-agnostic.
        """
        # `q_dtype` and `output_dtype` are independent comptime type
        # parameters so callers can pass TileTensors with literal dtypes
        # (e.g. `DType.float32` from `enqueue_create_buffer[DType.float32]`)
        # without Mojo failing to unify a literal with a generic
        # parameter (`Self.config.output_dtype`, `Self.config.dtype`) even
        # when they evaluate equal. The asserts pin the contract: q
        # matches `config.dtype` (BF16 or FP8), o matches
        # `config.output_dtype`. The downstream rebind is a no-op identity
        # given the assert.
        var _num_keys = Int(num_keys)
        var _start_pos = Int(start_pos)
        comptime assert (
            q_dtype == Self.config.dtype
        ), "MhaPrefillV2.run: `q.dtype` must equal `config.dtype`"
        comptime assert (
            output_dtype == Self.config.output_dtype
        ), "MhaPrefillV2.run: `o.dtype` must equal `config.output_dtype`"
        var q_bf16 = rebind[
            TileTensor[
                Self.config.dtype, q_layout, ImmutAnyOrigin, Engine=q_engine
            ]
        ](q)
        # `seq_len` from the layout's runtime dim and `num_tiles` from
        # the runtime `_num_keys` arg are wave-uniform by construction.
        # Wrapping in `readfirstlane` here — at the actual use site
        # inside the kernel — is what materializes the uniformity into
        # SGPR-resident operands across the main loop. Upstream
        # `readfirstlane` at the layout-construction site (as the
        # ragged kernel does) doesn't survive the TileTensor abstraction.
        var seq_len = Int(readfirstlane(Int32(q.dim[1]())))
        var num_tiles = Int(
            readfirstlane(Int32(ceildiv(_num_keys, Self.KV_BLOCK)))
        )
        comptime assert Self.NUM_HEADS % Self.NUM_KV_HEADS == 0, (
            "MhaPrefillV2: NUM_HEADS must be a multiple of NUM_KV_HEADS"
            " (GROUP = NUM_HEADS // NUM_KV_HEADS)"
        )
        # K/V SMEM byte layout: each slot is a 2D `(KV_BLOCK *
        # NUM_BLOCK_COLS, SUB_COLS)` row-major tile holding `SUB_ROWS x
        # SUB_COLS` sub-blocks linearized as
        # `subtile_id = block_row * (DEPTH / SUB_COLS) + block_col`.
        # `_dma_k` and `MhaMmaOp.load_V` both rely on this layout to
        # collapse subtile slicing to `tile[SUB_ROWS, SUB_COLS](id, 0)`.
        comptime _K_SUB_ROWS = Self._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._MmaOp.K_SUB_COLS
        comptime _V_SUB_ROWS = Self._MmaOp.V_SUB_ROWS
        comptime _V_SUB_COLS = Self._MmaOp.V_SUB_COLS
        comptime _NUM_BLOCK_COLS_K = Self.DEPTH // _K_SUB_COLS
        comptime _NUM_BLOCK_COLS_V = Self.DEPTH // _V_SUB_COLS
        comptime _K_SLOT_ROWS = Self.KV_BLOCK * _NUM_BLOCK_COLS_K
        comptime _V_SLOT_ROWS = Self.KV_BLOCK * _NUM_BLOCK_COLS_V
        comptime smem_layout_k = row_major[_K_SLOT_ROWS, _K_SUB_COLS]()
        comptime smem_layout_v = row_major[_V_SLOT_ROWS, _V_SUB_COLS]()

        # Four independent `smem_alloc` calls (one per double-buffer
        # slot). Letting the LDS allocator place each slot separately
        # yields wave-uniform base pointers that `readfirstlane` to
        # SGPR; the comptime `k_smem[0]` / `k_smem[1]` subscripts then
        # resolve to a fixed SGPR operand per call site. Consolidating
        # into one allocation + runtime `.tile[](stage, 0)` indexing
        # defeats that scalarization and regresses ~5-10%.
        #
        # Per-slot alignment = one sub-block. For K (BF16 32x32 = 2 KiB)
        # this keeps the two-XOR swizzle `bit5 ^= bit9; bit4 ^= bit10`
        # invariant across sub-blocks within a slot; V needs only the
        # 16-B `ds_read_b128` alignment.
        comptime _K_ALIGN = (
            _K_SUB_ROWS * _K_SUB_COLS * size_of[Self.config.dtype]()
        )
        comptime _V_ALIGN = (
            _V_SUB_ROWS * _V_SUB_COLS * size_of[Self.config.dtype]()
        )
        var k_smem_0_tt = smem_alloc[Self.config.dtype, alignment=_K_ALIGN](
            smem_layout_k
        )
        var k_smem_1_tt = smem_alloc[Self.config.dtype, alignment=_K_ALIGN](
            smem_layout_k
        )
        var v_smem_0_tt = smem_alloc[Self.config.dtype, alignment=_V_ALIGN](
            smem_layout_v
        )
        var v_smem_1_tt = smem_alloc[Self.config.dtype, alignment=_V_ALIGN](
            smem_layout_v
        )

        var k_smem = Tuple(k_smem_0_tt, k_smem_1_tt)
        var v_smem = Tuple(v_smem_0_tt, v_smem_1_tt)

        var w_id = Int(readfirstlane(warp_id()))
        var l_id = Int(lane_id())

        # GQA-aware head index: permutes block_idx.x ∈ [0, NUM_HEADS)
        # into head_idx ∈ [0, NUM_HEADS) so blocks visiting the same
        # KV head are spaced GROUP apart in launch order, spreading KV
        # bandwidth across CUs/XCDs. View block_x as a row-major index
        # into a (NUM_KV_HEADS, GROUP) rectangle (`bx_div ∈ [0,
        # NUM_KV_HEADS)` indexes the KV head, `bx_mod ∈ [0, GROUP)`
        # indexes the Q within that group); `head_idx = bx_mod *
        # NUM_KV_HEADS + bx_div` transposes to column-major over the
        # same rectangle. The map is a bijection iff NUM_HEADS ==
        # GROUP * NUM_KV_HEADS, which the divisibility assert above
        # enforces. Reduces to identity at MHA (GROUP=1) and at MQA
        # (NUM_KV_HEADS=1), keeping `divmod(_, 1)` constant-foldable
        # for byte-identical asm on those paths.
        var block_x = Int(readfirstlane(Int32(block_idx.x)))
        comptime _GROUP = Self.NUM_HEADS // Self.NUM_KV_HEADS
        var bx_div, bx_mod = divmod(block_x, _GROUP)
        var head_idx = bx_mod * Self.NUM_KV_HEADS + bx_div
        var block_tile_idx = Int(readfirstlane(Int32(block_idx.y)))
        var batch_idx = Int(readfirstlane(Int32(block_idx.z)))
        var kv_head_idx = head_idx // _GROUP
        var tile_idx = block_tile_idx * Self.NUM_WARPS + w_id

        # Lower-half warps `[0..NUM_WARPS/2)` vs upper-half — controls
        # the prologue/conclusion stagger barrier.
        var stagger = w_id >= (Self.NUM_WARPS // 2)

        # Causal cap on `max_num_tiles`: skip tiles whose K range is
        # entirely past the block's last Q row (every entry would be
        # masked anyway).
        var max_tile_idx_local = (
            block_tile_idx * Self.NUM_WARPS + Self.NUM_WARPS - 1
        )
        var max_q_end_pos = (
            max_tile_idx_local + 1
        ) * Self.Q_BLOCK_SIZE + _start_pos
        var max_num_tiles_calc = ceildiv(max_q_end_pos, Self.KV_BLOCK)
        var max_num_tiles_local: Int
        # FULL_MASK skip at the loop boundary for `CausalMask` (the
        # common case). For an arbitrary `MHAMask` we can't statically
        # bound the K iteration count, so we iterate to `num_tiles` and
        # rely on the per-tile `mask_functor.status(...)` check at each
        # mask site to detect a fully-masked tile and zero `att_block`.
        # `NullMask` keeps the full iteration count and `status` always
        # returns `NO_MASK`, so per-tile mask work is comptime-elided.
        comptime if mask_t == CausalMask:
            max_num_tiles_local = (
                max_num_tiles_calc if max_num_tiles_calc
                < num_tiles else num_tiles
            )
        elif mask_t == NullMask:
            # #87603: the software pipeline processes K tiles in even
            # pairs (main loop advances `j` by 2; epilogue drains 4). An
            # ODD tile count double-processes tile `N-3` across the
            # main-loop/epilogue boundary and corrupts the output (FLUX
            # i2i: _num_keys=8623 -> 135 tiles, odd). Round up to even with
            # a phantom trailing tile: the SRD clamp zeros its K/V and the
            # kbound mask excludes its score-0 columns, so it contributes
            # nothing. `CausalMask` is unaffected (its cap fixes the
            # parity); other masks keep the exact count.
            max_num_tiles_local = num_tiles + (num_tiles & 1)
        else:
            max_num_tiles_local = num_tiles

        # Per-(batch, head) 2D views via `.tile(Coord shape, Coord
        # indices).reshape(2D layout)`. The 4D `.tile` selects the
        # singleton `(1, seq_len, 1, DEPTH)` sub-tensor at
        # `(batch_idx, 0, head_idx, 0)`; `.reshape` retags it as a 2D
        # MixedLayout that the per-warp `.tile[Q_BLOCK_SIZE, DEPTH]()`
        # downstream and the DMA loaders expect.
        # Q/O batch indexing: non-ragged callers pass a multi-batch Q
        # tensor of shape `(batch_size, seq_len, num_heads, depth)`
        # and this kernel picks the per-batch slice via `batch_idx`. Ragged
        # callers pass a *per-sequence* Q tensor of shape `(1, seq_len,
        # num_heads, depth)` whose pointer is already pre-offset to
        # this sequence's slice in the packed buffer — for those,
        # indexing the batch dim by `batch_idx > 0` would OOB-read
        # into the next sequence's data. Force the batch coord to 0
        # for ragged so the per-sequence Q view is selected regardless
        # of `block_idx.z`. (FA2 sidesteps this by passing a raw
        # pointer instead of a TileTensor view.)
        var q_batch_coord = 0 if ragged else batch_idx
        var q_2d = q_bf16.tile(
            Coord(
                Idx[1],
                Int32(seq_len),
                Idx[1],
                Idx[Self.DEPTH],
            ),
            Coord(q_batch_coord, Idx[0], head_idx, Idx[0]),
        ).reshape(
            Self._QPerHeadLayoutT(
                Coord(
                    Int32(seq_len),
                    Idx[Self.DEPTH],
                ),
                Coord(Idx[Self._Q_ROW_STRIDE], Idx[1]),
            )
        )
        var o_2d = o.tile(
            Coord(
                Idx[1],
                Int32(seq_len),
                Idx[1],
                Idx[Self.DEPTH],
            ),
            Coord(q_batch_coord, Idx[0], head_idx, Idx[0]),
        ).reshape(
            Self._QPerHeadLayoutT(
                Coord(
                    Int32(seq_len),
                    Idx[Self.DEPTH],
                ),
                Coord(Idx[Self._Q_ROW_STRIDE], Idx[1]),
            )
        )
        var q_warp_block_idx = block_tile_idx * Self.NUM_WARPS + w_id
        var q_warp_2d = q_2d.tile[Self.Q_BLOCK_SIZE, Self.DEPTH](
            q_warp_block_idx, 0
        )
        var o_warp_2d = o_2d.tile[Self.Q_BLOCK_SIZE, Self.DEPTH](
            q_warp_block_idx, 0
        )

        # Per-(batch, kv_head) is captured via the MHAOperand interface.
        # `_dma_k` / `_dma_v` call `k.block_paged_tile[KV_BLOCK](...)` per
        # tile, returning a TileTensor anchored at `(batch_idx,
        # t*KV_BLOCK, kv_head_idx, 0)`. For LayoutTensorMHAOperand this
        # resolves to a pointer-arithmetic offset (contiguous K/V);
        # for KVCacheMHAOperand it resolves through the page table.
        # Cast indices to UInt32 once so the trait calls don't repeat
        # the conversion.
        var _batch_idx_u32 = UInt32(batch_idx)
        var _kv_head_idx_u32 = UInt32(kv_head_idx)

        # Q load + prescale by `scale * log2(e)`; multiply in FP32
        # per fragment to preserve ~1 ULP, cast back to BF16.
        var scale_log2e = scale * 1.4426950408889634
        var q_reg = Self._load_q_and_scale(q_warp_2d, scale_log2e)

        # Persistent kernel-scope state. `scale_vec` initialized to ones
        # so the epilogue's unconditional `norm_vec *= scale_vec` is a
        # safe no-op when no rescale ever fired.
        #
        # Sink path: pre-seed the online-softmax recurrence so the hot
        # loop stays sink-agnostic. The AMD convention is to encode the
        # virtual sink token as the initial `(max_vec, norm_vec)`
        # state. `max_vec_init = log2e * sink_weight[head_idx]` keeps
        # the rowmax in log2 units (Q is already prescaled by `scale *
        # log2e`, so att values are in log2 units). `norm_vec_init = 1`
        # reflects the virtual sink's
        # `exp2(score - max) = exp2(0) = 1` contribution. Subsequent
        # tiles update through the normal recurrence; the sink is
        # rescaled implicitly as the running max grows.
        var o_reg = reg_alloc[.float32](Self._MmaOp.O_LAYOUT)
        # `OnlineSoftmax` owns the 4 row-state scalars (`max_vec`,
        # `max_vec_prev`, `norm_vec`, `scale_vec`) as `Float32` fields
        # — 1 VGPR/lane each.
        #
        # CORRECTNESS NOTE: construct via `OnlineSoftmax()` directly
        # (the canonical Mojo idiom that runs the constructor on the
        # new value). The previous `stack_allocation[1, OnlineSoftmax,
        # LOCAL]()[]` + `softmax.__init__()` pattern — which was
        # adopted to coax LLVM into slotting the 4 scalars next to the
        # other LOCAL-address-space allocas for tighter live-range
        # sharing — silently produced a uniform `+185` (≈ log2e ×
        # DEPTH) leak in `norm_vec` for the BF16 CausalMask d=128 path
        # and dropped this kernel's vs-naive cos_sim to ~0.78. The leak only
        # manifests when `max_num_tiles_local` is small enough that
        # the main loop is skipped (CausalMask at d=128 hits
        # `max_num_tiles_local = 4`, main loop condition
        # `j=3 < N-1=3` is false). Likely root cause: invoking
        # `__init__(out self)` on an already-bound `softmax` local
        # rebinds the SSA reads of the four fields to the freshly
        # initialized values but leaves the alloca uninitialized;
        # cluster fns that mutate `softmax` via `mut self` then write
        # back to the alloca rather than the SSA local, and the next
        # read picks up the uninitialized memory. A short main loop
        # exposes the bug because no rescale fires to overwrite the
        # garbage; the long-loop NullMask/Rescale paths happen to
        # rewrite the state before the epilogue reads it.
        var softmax = OnlineSoftmax[Self._SOFTMAX_DTYPE]()

        # Mask functor bundle: comptime block sizes baked in so the
        # dispatch arithmetic folds to literals.
        var mask = MaskApplier[
            Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE, KV_BLOCK_SIZE=Self.KV_BLOCK
        ](mask_functor)
        _ = o_reg.fill(0)
        comptime if sink:
            # `log2e` baked in at init so the recurrence sees a rowmax
            # already in log2 units. `sink_weights_ptr[head_idx]` is
            # per-q-head (gpt-oss style); read once per block, broadcast
            # to the single-element row state. `1.4426950408889634` is
            # `log2(e)` (same constant used for Q prescale at line
            # `scale_log2e = scale * 1.4426950408889634`).
            var sw_raw = sink_weights_ptr[head_idx]
            var sw_log2 = sw_raw.cast[.float32]() * 1.4426950408889634
            softmax.reseed_with_sink(sw_log2)

        # Two `att_block` slots; the loop ping-pongs which one is the
        # QK destination and which the softmax source. Element dtype
        # is `Self._SOFTMAX_DTYPE` — FP32 for the BF16 attention path
        # (precision preserved across the rescale recurrence) and
        # BF16 for the FP8 attention path (sub-step 8: halves the
        # tile's VGPR footprint to make Phase 7.2's KV=128 fit under
        # the 128 VGPR/thread cap).
        var att_block_0 = reg_alloc[Self._SOFTMAX_DTYPE](Self._MmaOp.ATT_LAYOUT)
        var att_block_1 = reg_alloc[Self._SOFTMAX_DTYPE](Self._MmaOp.ATT_LAYOUT)

        # Persistent BF16 P-cache shared across all six `_att_bf16_full`
        # producers and their PV consumers. The persistent destination
        # keeps the cast from rematerializing at each PV use site.
        var att_block_bf16 = reg_alloc[Self.config.dtype](
            Self._MmaOp.ATT_BF16_FULL_LAYOUT
        )

        # Persistent `k_reg`: K is loaded once per K-tile in a dedicated
        # cluster (C3/C7) and consumed in the next QK cluster (C0/C4)
        # without any in-cluster `ds_read`s.
        var k_reg = reg_alloc[Self.config.dtype](Self._MmaOp.K_LAYOUT)

        # === Prologue ===
        # K[0] DMA, full drain, barrier.
        Self._dma_k(
            k_smem[0],
            k,
            _batch_idx_u32,
            _kv_head_idx_u32,
            0,
            _num_keys,
            w_id,
            l_id,
        )
        s_waitcnt[vmcnt=UInt32(0), lgkmcnt=UInt32(0)]()
        _sched_barrier_zero()
        _s_barrier_raw()

        # K[1] and V[0] DMAs.
        if num_tiles > 1:
            Self._dma_k(
                k_smem[1],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                1,
                _num_keys,
                w_id,
                l_id,
            )
        Self._dma_v(
            v_smem[0],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            0,
            _num_keys,
            w_id,
            l_id,
        )
        _sched_barrier_zero()
        _s_barrier_raw()

        Self._load_k_reg(k_reg, k_smem[0])

        # This `lgkmcnt(0)` is semantically redundant after the barrier
        # above, but removing it costs ~1.1% — likely acting as a
        # scheduler-region anchor that helps RA. Keep until understood.
        s_waitcnt[lgkmcnt=UInt32(0)]()

        # QK[0] over pre-loaded `k_reg`.
        Self._qk_with_kreg(att_block_0, k_reg, q_reg, scale_log2e)
        _sched_barrier_zero()

        # Prologue tile 0 mask.
        mask.apply(
            att_block_0,
            tile_idx,
            0,
            _start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
            _num_keys,
        )

        # Tile-0 partial softmax (no rescale: first tile).
        softmax.seed_tile0(att_block_0)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_0)
        _sched_barrier_zero()

        # Stagger barrier — upper-half warps wait.
        if stagger:
            _sched_barrier_zero()
            _s_barrier_raw()

        # Pre-load K[1] ahead of main loop's first C0.
        if num_tiles > 1:
            Self._load_k_reg(k_reg, k_smem[1])

        # K[2], V[1] DMAs.
        if num_tiles > 2:
            Self._dma_k(
                k_smem[0],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                2,
                _num_keys,
                w_id,
                l_id,
            )
        if num_tiles > 1:
            Self._dma_v(
                v_smem[1],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                1,
                _num_keys,
                w_id,
                l_id,
            )
        _sched_barrier_zero()
        _s_barrier_raw()

        var pending_scale: Bool = False

        # === Main loop ===
        # 8 clusters per iteration, advancing `j` by 2 each pass.
        var j: Int = 3
        while j < max_num_tiles_local - 1:
            # C0: QK[j-2] + tail softmax of tile (j-3); bulk BF16 cast
            # into `att_block_bf16` happens inside the helper, pre-cast
            # for C2's PV.
            _asm_label["; MHA_MAIN_C0_BEGIN"]()
            Self._qk_tail_softmax_cluster[
                sched_group=Self._SCHED_MAIN_C0_QK_TAIL
            ](
                att_block_1,
                att_block_0,
                att_block_bf16,
                k_reg,
                q_reg,
                softmax,
                pending_scale,
                scale_log2e,
            )
            _cluster_barrier()

            # C1: DMA K[j], load v_reg = V[j-3]. For non-Causal masks
            # also apply the mask to `att_block_1` (= QK[j-2] written
            # in C0) before C2's partial softmax reads it. The C0/C2
            # mask is comptime-elided for `CausalMask`: the
            # `max_num_tiles` cap already guarantees tile (j-2) is
            # naturally fully unmasked. Non-causal masks
            # (SlidingWindow / Chunked) need the explicit call —
            # otherwise unmasked Q@K values bleed into tile (j-2)'s
            # softmax.
            _asm_label["; MHA_MAIN_C1_BEGIN"]()
            Self._dma_k(
                k_smem[1],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j,
                _num_keys,
                w_id,
                l_id,
            )
            var v_reg_c1 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
            Self._load_v_reg(v_reg_c1, v_smem[0])
            comptime if mask_t != CausalMask:
                mask.apply(
                    att_block_1,
                    tile_idx,
                    j - 2,
                    _start_pos,
                    UInt32(head_idx),
                    _batch_idx_u32,
                    l_id,
                    _num_keys,
                )
            _cluster_barrier()

            # C2: PV[j-3] strip-interleaved with partial softmax of
            # tile (j-2), consuming `att_block_bf16` pre-cast in C0.
            _asm_label["; MHA_MAIN_C2_BEGIN"]()
            pending_scale = Self._pv_strip_with_partial_softmax[
                sched_group=Self._SCHED_MAIN_C2_PV_PARTIAL
            ](
                v_reg_c1,
                att_block_bf16,
                o_reg,
                softmax,
                att_block_1,
            )
            _cluster_barrier()

            # C3: DMA V[j-1], pre-load k_reg = K[j-1] for the next C4.
            _asm_label["; MHA_MAIN_C3_BEGIN"]()
            Self._dma_v(
                v_smem[0],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j - 1,
                _num_keys,
                w_id,
                l_id,
            )
            Self._load_k_reg(k_reg, k_smem[0])
            _cluster_barrier()

            # C4: QK[j-1] + tail softmax of tile (j-2); bulk cast into
            # `att_block_bf16` pre-stages C6's PV-A.
            _asm_label["; MHA_MAIN_C4_BEGIN"]()
            Self._qk_tail_softmax_cluster[
                sched_group=Self._SCHED_MAIN_C4_QK_TAIL
            ](
                att_block_0,
                att_block_1,
                att_block_bf16,
                k_reg,
                q_reg,
                softmax,
                pending_scale,
                scale_log2e,
            )
            _cluster_barrier()

            # C5: DMA K[j+1], load v_reg = V[j-2], apply mask to
            # att_block_0 for tile (j-1). No MFMAs — IGLP interleaves
            # the 32 V `ds_read`s with the mask's VALU pairs
            # (`v_cmp` / `v_cndmask` for the CausalMask fast path;
            # per-element loop for the generic path).
            _asm_label["; MHA_MAIN_C5_BEGIN"]()
            Self._dma_k(
                k_smem[0],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j + 1,
                _num_keys,
                w_id,
                l_id,
            )
            var v_reg_c5 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
            Self._load_v_reg(v_reg_c5, v_smem[1])
            mask.apply(
                att_block_0,
                tile_idx,
                j - 1,
                _start_pos,
                UInt32(head_idx),
                _batch_idx_u32,
                l_id,
                _num_keys,
            )
            sched_dsread_valu_pairs[
                32, valu_cnt=1, group=Self._SCHED_MAIN_C5_DSREAD
            ]()
            _cluster_barrier()

            # C6: PV[j-2] strip-interleaved with partial softmax of
            # tile (j-1), consuming `att_block_bf16` pre-cast in C4.
            _asm_label["; MHA_MAIN_C6_BEGIN"]()
            pending_scale = Self._pv_strip_with_partial_softmax[
                sched_group=Self._SCHED_MAIN_C6_PV_PARTIAL
            ](
                v_reg_c5,
                att_block_bf16,
                o_reg,
                softmax,
                att_block_0,
            )
            _cluster_barrier()

            # C7: DMA V[j], pre-load k_reg = K[j] for the next iter's C0.
            _asm_label["; MHA_MAIN_C7_BEGIN"]()
            Self._dma_v(
                v_smem[1],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j,
                _num_keys,
                w_id,
                l_id,
            )
            Self._load_k_reg(k_reg, k_smem[1])
            _cluster_barrier()

            j += 2

        # === Epilogue ===
        # 13 clusters draining the final 4 tiles `N-4..N-1`; assumes
        # `num_tiles >= 4`.
        var N = max_num_tiles_local

        # Epi-C0: QK[N-3] + tail softmax of tile (N-4). Tail rescale
        # is UNCONDITIONAL; `scale_vec` initialized to ones makes it
        # an identity when no rescale ever fired.
        _asm_label["; MHA_EPI_C0_BEGIN"]()
        Self._qk_with_kreg(att_block_1, k_reg, q_reg, scale_log2e)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C0_TAIL](
            att_block_0,
            softmax,
        )
        # Pre-cast for Epi-C2's PV.
        Self._att_bf16_full(att_block_bf16, att_block_0)
        _cluster_barrier()

        # Epi-C1: DMA K[N-1], load v_reg = V[N-4], apply mask to
        # att_block_1 for tile (N-3).
        _asm_label["; MHA_EPI_C1_BEGIN"]()
        Self._dma_k(
            k_smem[1],
            k,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 1,
            _num_keys,
            w_id,
            l_id,
        )
        var v_reg_e1 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e1, v_smem[0])
        mask.apply(
            att_block_1,
            tile_idx,
            N - 3,
            _start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
            _num_keys,
        )
        _cluster_barrier()

        # Epi-C2: PV[N-4] (whole) + partial softmax of tile (N-3),
        # consuming `att_block_bf16` pre-cast in Epi-C0.
        _asm_label["; MHA_EPI_C2_BEGIN"]()
        Self._pv_whole_with_partial_softmax[
            sched_group=Self._SCHED_EPI_C2_PV_PARTIAL
        ](
            v_reg_e1,
            att_block_bf16,
            o_reg,
            softmax,
            att_block_1,
        )
        _cluster_barrier()

        # Epi-C3: DMA V[N-2], pre-load k_reg = K[N-2].
        _asm_label["; MHA_EPI_C3_BEGIN"]()
        Self._dma_v(
            v_smem[0],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 2,
            _num_keys,
            w_id,
            l_id,
        )
        Self._load_k_reg(k_reg, k_smem[0])
        _cluster_barrier()

        # Epi-C4: QK[N-2] + tail softmax of tile (N-3); pre-cast for
        # Epi-C6's PV.
        _asm_label["; MHA_EPI_C4_BEGIN"]()
        Self._qk_with_kreg(att_block_0, k_reg, q_reg, scale_log2e)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C4_TAIL](
            att_block_1,
            softmax,
        )
        Self._att_bf16_full(att_block_bf16, att_block_1)
        _cluster_barrier()

        # Epi-C5: load v_reg = V[N-3], apply mask to att_block_0
        # for tile (N-2).
        _asm_label["; MHA_EPI_C5_BEGIN"]()
        var v_reg_e5 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e5, v_smem[1])
        mask.apply(
            att_block_0,
            tile_idx,
            N - 2,
            _start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
            _num_keys,
        )
        sched_dsread_valu_pairs[
            32, valu_cnt=1, group=Self._SCHED_EPI_C5_DSREAD
        ]()
        _cluster_barrier()

        # Epi-C6: PV[N-3] (whole) + partial softmax of tile (N-2),
        # consuming `att_block_bf16` pre-cast in Epi-C4.
        _asm_label["; MHA_EPI_C6_BEGIN"]()
        Self._pv_whole_with_partial_softmax[
            sched_group=Self._SCHED_EPI_C6_PV_PARTIAL
        ](
            v_reg_e5,
            att_block_bf16,
            o_reg,
            softmax,
            att_block_0,
        )
        _cluster_barrier()

        # Epi-C7: DMA V[N-1], pre-load k_reg = K[N-1].
        _asm_label["; MHA_EPI_C7_BEGIN"]()
        Self._dma_v(
            v_smem[1],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 1,
            _num_keys,
            w_id,
            l_id,
        )
        Self._load_k_reg(k_reg, k_smem[1])
        _cluster_barrier()

        # Epi-C8: QK[N-1] + tail softmax of tile (N-2); pre-cast for
        # Epi-C10's PV.
        _asm_label["; MHA_EPI_C8_BEGIN"]()
        Self._qk_with_kreg(att_block_1, k_reg, q_reg, scale_log2e)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C8_TAIL](
            att_block_0,
            softmax,
        )
        Self._att_bf16_full(att_block_bf16, att_block_0)
        _cluster_barrier()

        # Epi-C9: load v_reg = V[N-2], apply mask to att_block_1
        # for tile (N-1).
        _asm_label["; MHA_EPI_C9_BEGIN"]()
        var v_reg_e9 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e9, v_smem[0])
        mask.apply(
            att_block_1,
            tile_idx,
            N - 1,
            _start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
            _num_keys,
        )
        sched_dsread_valu_pairs[
            32, valu_cnt=1, group=Self._SCHED_EPI_C9_DSREAD
        ]()
        _cluster_barrier()

        # Epi-C10: PV[N-2] (whole) + FULL softmax of tile (N-1) +
        # final `o_reg *= scale_vec`. The PV-A read happens before the
        # next `_att_bf16_full` overwrites the cache. Pre-casts for
        # Epi-C12's PV.
        _asm_label["; MHA_EPI_C10_BEGIN"]()
        Self._pv_whole(v_reg_e9, att_block_bf16, o_reg)
        Self._full_softmax_unconditional[sched_group=Self._SCHED_EPI_C10_FULL](
            att_block_1,
            softmax,
        )
        Self._att_bf16_full(att_block_bf16, att_block_1)
        _sched_barrier_zero()
        softmax.rescale_output(o_reg)
        _s_barrier_raw()
        _sched_barrier_zero()

        # Epi-C11: load v_reg = V[N-1].
        _asm_label["; MHA_EPI_C11_BEGIN"]()
        var v_reg_e11 = reg_alloc[Self.config.dtype](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e11, v_smem[1])
        _cluster_barrier()

        # Epi-C12: PV[N-1] (whole) over `att_block_bf16` pre-cast in
        # Epi-C10, then final `o_reg /= norm_vec` in-place.
        #
        # In-place divide avoids materializing a second FP32 O_LAYOUT
        # tile (64 VGPRs/lane) alongside `o_reg`. At FP8 KV=128 the
        # combined live set hit the 128 VGPR/thread cap and spilled
        # 9 VGPR-equivalents to scratch in this exact cluster
        # (rocprof + EMIT_ASM_LABELS confirmed all `scratch_*` ops in
        # the kernel live in `MHA_EPI_C12`).
        _asm_label["; MHA_EPI_C12_BEGIN"]()
        Self._pv_whole(v_reg_e11, att_block_bf16, o_reg)
        softmax.normalize_output(o_reg)
        _cluster_barrier()

        # Conclusion barrier — lower-half warps wait.
        if not stagger:
            _s_barrier_raw()

        # Output store. col_l → row_l is a zero-cost re-tag of the same
        # per-lane storage, so we construct a TileTensor view over
        # `o_reg.ptr` directly under `O_T_LAYOUT` instead of invoking a
        # transpose primitive. The in-place divide above made `o_reg`
        # hold the normalized output.
        comptime _o_view_layout = Self._MmaOp.O_T_LAYOUT
        var o_normalized_view = TileTensor[
            .float32,
            type_of(_o_view_layout),
            MutUntrackedOrigin,
            address_space=.LOCAL,
        ](o_reg.ptr, _o_view_layout)
        var epilogue_writer = RegTileEpilogue[output_dtype, 1](o_warp_2d)
        # Partial-Q-tile bound: for sequences whose length is not a
        # multiple of BM, the last tile owns fewer than BM valid Q rows
        # (OOB Q rows read 0 from buffer_load via RegTileLoader and
        # compute a finite-but-meaningless output). The writeback is
        # the only place that has to gate — RegTileEpilogue leaves the
        # M check to the caller.
        var valid_q_rows = max(
            0, min(Self.BM, seq_len - block_tile_idx * Self.BM)
        )
        var valid_q_rows_in_warp = min(
            Self.Q_BLOCK_SIZE,
            max(0, valid_q_rows - w_id * Self.Q_BLOCK_SIZE),
        )
        Self._store_o_to_gmem[output_dtype](
            o_normalized_view, epilogue_writer, l_id, valid_q_rows_in_warp
        )

    # ===-------------------------------------------------------------=== #
    # Ragged-batch GPU entry point. Wraps `run` with per-sequence setup
    # (start_of_seq / q_batch_offset / seq_len / num_keys / start_pos)
    # so the dispatcher can launch one grid for a packed multi-sequence
    # batch. Grid: `(NUM_HEADS, ceildiv(max_prompt_len, BM), batch_size)`.
    # ===-------------------------------------------------------------=== #
    @__llvm_metadata(`llvm.amdgpu-waves-per-eu`=__mlir_attr.`"2,2"`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @__name(
        t"mha_prefill_v2_ragged_amd_{qkv_dtype}_{output_dtype}_BM{Self.BM}_KV{Self.KV_BLOCK}_D{Self.DEPTH}"
    )
    @staticmethod
    def ragged_kernel[
        k_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        qkv_dtype: DType,
        output_dtype: DType,
        cross_attention: Bool = False,
        sink: Bool = False,
    ](
        q_ptr: UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin],
        k: k_t,
        v: v_t,
        output_ptr: UnsafePointer[Scalar[output_dtype], MutAnyOrigin],
        mask_functor: mask_t,
        scale: Float32,
        input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
        kv_input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
        sink_weights_ptr: UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin],
    ):
        """Ragged-batch GPU kernel entry: per-sequence setup + `run`.

        The non-ragged equivalent is `run` itself (which takes
        already-sliced per-batch TileTensors). For ragged, this
        wrapper does the per-block ragged setup so the launcher can
        pass a single packed Q pointer + `input_row_offsets`.

        `cross_attention=False` (default): self-attention, where K/V
        length equals Q length plus any cached prefix. `num_keys`
        derives from `start_pos + seq_len`. `kv_input_row_offsets_ptr`
        is unused (caller may pass any well-typed stub).

        `cross_attention=True`: encoder-decoder style. K/V lengths come
        from `kv_input_row_offsets_ptr`, independent of the Q-side
        offsets. Mirrors the FA2 contract at `mha.mojo:1755-1762`.

        Parameters:
            k_t: K operand type (inferred); any `MHAOperand` whose
                `block_paged_tile` returns `(KV_BLOCK, DEPTH)` tiles.
            v_t: V operand type (inferred); same tile contract as
                `k_t`.
            mask_t: Mask functor type (inferred); selects the
                per-tile masking strategy.
            qkv_dtype: Element dtype of Q, K, and V (inferred);
                must equal `config.dtype`.
            output_dtype: Element dtype of the output buffer
                (inferred); must equal `config.output_dtype`.
            cross_attention: Whether K/V length is independent of Q
                (encoder-decoder style) (defaults to False).
            sink: Whether to seed the online softmax with
                attention-sink weights (defaults to False).

        Args:
            q_ptr: Pointer to the packed Q buffer, pre-offset per
                sequence by `input_row_offsets_ptr`.
            k: K operand (`MHAOperand`).
            v: V operand (`MHAOperand`).
            output_ptr: Pointer to the output buffer, same packing
                as `q_ptr`.
            mask_functor: Per-tile mask predicate (causal,
                sliding-window, etc.).
            scale: Softmax scale (`1/sqrt(DEPTH)`).
            input_row_offsets_ptr: Cumulative uint32 row offsets for
                Q sequences, length `batch_size + 1`.
            kv_input_row_offsets_ptr: Cumulative uint32 row offsets
                for K/V sequences; read only when `cross_attention`
                is True.
            sink_weights_ptr: Per-q-head attention-sink scalar
                weights; read only when `sink` is True.
        """
        # Wave-uniform prologue. Values are uniform by construction
        # (one read per block, broadcast). The uniformity hint that
        # actually matters is the `readfirstlane` inside `run` on
        # `seq_len` / `num_tiles` — at the use site inside the main
        # loop. Wraps here at the construction site don't survive the
        # TileTensor layout abstraction; ISA + bench confirmed no
        # effect.
        var batch_idx = block_idx.z
        var start_of_seq = Int(input_row_offsets_ptr[batch_idx])
        var end_of_seq = Int(input_row_offsets_ptr[batch_idx + 1])
        var seq_len = end_of_seq - start_of_seq

        # Out-of-range block guard: grid is sized by max_prompt_len;
        # if this sequence is shorter, the trailing q-blocks have
        # nothing to do.
        if Int(block_idx.y) * Self.BM >= seq_len:
            return

        var start_pos = Int(k.cache_length(batch_idx))
        var num_keys: Int
        comptime if cross_attention:
            # Encoder/decoder cross-attention: K/V length is independent
            # of the Q-side seq_len and comes from a separate
            # kv-side input_row_offsets table.
            var kv_start = Int(kv_input_row_offsets_ptr[batch_idx])
            var kv_end = Int(kv_input_row_offsets_ptr[batch_idx + 1])
            num_keys = (kv_end - kv_start) + start_pos
        else:
            num_keys = start_pos + seq_len

        var q_batch_offset = (
            start_of_seq * Self.config.num_heads * Self.config.depth
        )

        # E_B: share the rank-4 BSHD layout between q_tt and o_tt so
        # the runtime seq_len field lives once. Halves the per-block
        # live state introduced by the layout construction.
        var ragged_layout = row_major(
            Coord(
                1,
                seq_len,
                Self.config.num_heads,
                Self.config.depth,
            )
        )
        var q_tt = TileTensor(q_ptr + q_batch_offset, ragged_layout)
        var o_tt = TileTensor(output_ptr + q_batch_offset, ragged_layout)

        Self.run[
            k_t,
            v_t,
            mask_t,
            qkv_dtype,
            output_dtype,
            type_of(q_tt).LayoutType,
            type_of(o_tt).LayoutType,
            type_of(q_tt).Engine,
            type_of(o_tt).Engine,
            ragged=True,
            sink=sink,
        ](
            q_tt,
            k,
            v,
            o_tt,
            mask_functor,
            scale,
            Int32(num_keys),
            Int32(start_pos),
            sink_weights_ptr,
        )


@always_inline
def mha_prefill_v2_ragged[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    qkv_dtype: DType,
    output_dtype: DType,
    //,
    config: MhaConfigV2,
    cross_attention: Bool = False,
    sink: Bool = False,
    compile_options: StaticString = CompilationTarget[
        DeviceContext.default_device_info.target()
    ].default_compile_options(),
](
    q_ptr: UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_dtype], MutAnyOrigin],
    mask_functor: mask_t,
    scale: Float32,
    input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
    kv_input_row_offsets_ptr: UnsafePointer[UInt32, ImmutAnyOrigin],
    max_prompt_len: Int,
    batch_size: Int,
    ctx: DeviceContext,
    sink_weights_ptr: UnsafePointer[
        Scalar[qkv_dtype], ImmutAnyOrigin
    ] = UnsafePointer[Scalar[qkv_dtype], ImmutAnyOrigin].unsafe_dangling(),
) raises:
    """Host launcher for ragged `MhaPrefillV2` prefill.

    Mirrors `mha_prefill_v2` for non-ragged, but each block does the
    per-sequence ragged setup (start_of_seq / q_batch_offset / seq_len /
    num_keys / start_pos) and constructs its rank-4 BSHD Q/O view
    internally, so the caller doesn't have to pre-slice per sequence.

    `cross_attention=False` (default): self-attention. K/V length per
    batch derives from `start_pos + (end_of_seq - start_of_seq)`.
    `kv_input_row_offsets_ptr` is unused inside the kernel and the
    caller may pass any well-typed stub (the dispatcher passes
    `input_row_offsets_ptr` itself).

    `cross_attention=True`: encoder-decoder. Pass the kv-side
    `input_row_offsets` (uint32 cumulative sum, length `batch_size + 1`).

    Grid: `(NUM_HEADS, ceildiv(max_prompt_len, BM), batch_size)`. Blocks
    where `block_idx.y * BM >= seq_len` for this sequence early-return.
    Partial-Q-tile (`seq_len % BM != 0`) is handled internally via
    lane-gated `_store_o_to_gmem`.

    Parameters:
        k_t: K operand type (inferred); any `MHAOperand` whose
            `block_paged_tile` returns `(KV_BLOCK, DEPTH)` tiles.
        v_t: V operand type (inferred); same tile contract as `k_t`.
        mask_t: Mask functor type (inferred); selects the per-tile
            masking strategy (causal, sliding-window, null, etc.).
        qkv_dtype: Element dtype of Q, K, and V (inferred); must equal
            `config.dtype`.
        output_dtype: Element dtype of the output buffer (inferred);
            must equal `config.output_dtype`.
        config: Shape configuration (`MhaConfigV2`).
        cross_attention: Whether K/V length is independent of Q
            (encoder-decoder style) (defaults to False).
        sink: Whether to seed the online softmax with attention-sink
            weights (defaults to False).
        compile_options: LLVM compile options string forwarded to
            `ctx.compile_function` (defaults to the device's
            `default_compile_options`).

    Args:
        q_ptr: Pointer to the packed Q buffer, pre-offset per sequence
            by `input_row_offsets_ptr`.
        k: K operand (`MHAOperand`).
        v: V operand (`MHAOperand`).
        output_ptr: Pointer to the output buffer, same packing as
            `q_ptr`.
        mask_functor: Per-tile mask predicate (causal, sliding-window,
            etc.).
        scale: Softmax scale (`1/sqrt(DEPTH)`).
        input_row_offsets_ptr: Cumulative uint32 row offsets for Q
            sequences, length `batch_size + 1`.
        kv_input_row_offsets_ptr: Cumulative uint32 row offsets for
            K/V sequences; read only when `cross_attention` is True.
        max_prompt_len: Maximum sequence length across the batch;
            sizes the grid's `block_idx.y` dimension.
        batch_size: Number of sequences in the packed batch; sizes the
            grid's `block_idx.z` dimension.
        ctx: Device context used to compile and enqueue the kernel.
        sink_weights_ptr: Per-q-head attention-sink scalar weights;
            read only when `sink` is True. Callers may pass
            `unsafe_dangling()` when `sink=False`.
    """
    comptime assert (
        qkv_dtype == config.dtype
    ), "mha_prefill_v2_ragged: `qkv_dtype` must equal `config.dtype`"
    comptime kernel = MhaPrefillV2[config].ragged_kernel[
        k_t, v_t, mask_t, qkv_dtype, output_dtype, cross_attention, sink
    ]
    var compiled = ctx.compile_function[
        kernel, compile_options=compile_options
    ]()
    comptime BM = MhaPrefillV2[config].BM
    ctx.enqueue_function(
        compiled,
        q_ptr,
        k,
        v,
        output_ptr,
        mask_functor,
        scale,
        input_row_offsets_ptr,
        kv_input_row_offsets_ptr,
        sink_weights_ptr,
        grid_dim=(
            config.num_heads,
            ceildiv(max_prompt_len, BM),
            batch_size,
        ),
        block_dim=MhaPrefillV2[config].NUM_THREADS,
    )


@always_inline
def mha_prefill_v2[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MhaConfigV2,
    sink: Bool = False,
    compile_options: StaticString = CompilationTarget[
        DeviceContext.default_device_info.target()
    ].default_compile_options(),
](
    q: TileTensor[mut=False, ...],
    k: k_t,
    v: v_t,
    o: TileTensor[mut=True, ...],
    mask_functor: mask_t,
    scale: Float32,
    num_keys: Int,
    start_pos: Int,
    ctx: DeviceContext,
    sink_weights_ptr: UnsafePointer[
        Scalar[q.dtype], ImmutAnyOrigin
    ] = UnsafePointer[Scalar[q.dtype], ImmutAnyOrigin].unsafe_dangling(),
) raises:
    """Host launcher for `MhaPrefillV2`.

    Derives grid dimensions from the Q layout and `num_keys`, compiles
    `MhaPrefillV2[config].run` (with optional caller-supplied LLVM
    `compile_options` such as `amdgpu-igrouplp-exact-solver=true` for
    benchmarks), and enqueues it.

    - `q`, `o`: `(batch, seq_len, num_heads, depth)` TileTensor.
      `o`'s dtype matches `config.output_dtype`: BF16 for production
      inference (which the dispatcher uses) or FP32 if the caller wants
      the unnormalized accumulator.
    - `k`, `v`: any `MHAOperand` (`LayoutTensorMHAOperand` for tests/
      bench, `KVCacheMHAOperand` for paged production caches at
      `page_size >= KV_BLOCK = 64`).

    `batch` and `seq_len` / `num_keys` may be dynamic; the head and
    depth dims must be static.

    Parameters:
        k_t: K operand type (inferred); any `MHAOperand` whose
            `block_paged_tile` returns `(KV_BLOCK, DEPTH)` tiles.
        v_t: V operand type (inferred); same tile contract as `k_t`.
        mask_t: Mask functor type (inferred); selects the per-tile
            masking strategy (causal, sliding-window, null, etc.).
        config: Shape configuration (`MhaConfigV2`).
        sink: Whether to seed the online softmax with attention-sink
            weights (defaults to False).
        compile_options: LLVM compile options string forwarded to
            `ctx.compile_function` (defaults to the device's
            `default_compile_options`).

    Args:
        q: Q tile tensor of shape `(batch, seq_len, num_heads, depth)`;
            dtype must equal `config.dtype`.
        k: K operand (`MHAOperand`).
        v: V operand (`MHAOperand`).
        o: Output tile tensor of shape `(batch, seq_len, num_heads,
            depth)`; dtype must equal `config.output_dtype`.
        mask_functor: Per-tile mask predicate (causal, sliding-window,
            etc.).
        scale: Softmax scale (`1/sqrt(DEPTH)`).
        num_keys: Runtime length of the K/V sequence.
        start_pos: Position of the first Q row in the global sequence;
            non-zero for prefill chunks of a longer generation.
        ctx: Device context used to compile and enqueue the kernel.
        sink_weights_ptr: Per-q-head attention-sink scalar weights;
            read only when `sink` is True. Callers may pass
            `unsafe_dangling()` when `sink=False`.
    """
    # Operand dtypes are taken dtype-generic at the launcher boundary
    # because Mojo doesn't unify a caller-site literal (e.g.
    # `DType.float32` from `enqueue_create_buffer[DType.float32]`) with
    # a generic comptime parameter (`config.output_dtype`,
    # `config.dtype`) even when they evaluate equal. Asserting
    # explicitly here gives the same safety without forcing
    # `rebind[TileTensor[config.dtype, ...]](q)` at every call site
    # (tests, bench, and the production dispatcher).
    comptime assert (
        q.dtype == config.dtype
    ), "mha_prefill_v2: `q.dtype` must equal `config.dtype`"
    comptime assert (
        o.dtype == config.output_dtype
    ), "mha_prefill_v2: `o.dtype` must equal `config.output_dtype`"

    var batch = Int(q.dim[0]())
    var seq_len = Int(q.dim[1]())

    comptime kernel = MhaPrefillV2[config]
    comptime kernel_run = kernel.run[
        k_t,
        v_t,
        mask_t,
        q.dtype,
        o.dtype,
        q.LayoutType,
        o.LayoutType,
        q.Engine,
        o.Engine,
        ragged=False,
        sink=sink,
    ]

    var compiled = ctx.compile_function[
        kernel_run, compile_options=compile_options
    ]()
    ctx.enqueue_function(
        compiled,
        q,
        k,
        v,
        o,
        mask_functor,
        scale,
        Int32(num_keys),
        Int32(start_pos),
        sink_weights_ptr,
        grid_dim=(
            config.num_heads,
            ceildiv(seq_len, kernel.BM),
            batch,
        ),
        block_dim=kernel.NUM_THREADS,
    )
