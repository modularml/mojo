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
"""SM100 (B200) warp-specialized PREFILL variant of the FP8 MLA indexer scorer.

Computes the identical per-(query token, key) logit as the shipped scorer
(`sparse_index_fp8_sm100.fp8_index_score_sm100`) and the scalar
`nn.index_fp8.fp8_index_kernel`:

    score[token, key] = k_scale[key]
                        * Σ_head relu(q[token, head] · k[key]) * q_scale[token, head]

The shipped kernel is K-resident / Q-streaming: one CTA holds a `BM_key`-key tile
and streams every query token past it. That maps a batch-1 prefill onto only
`num_keys / BM_key` CTAs and runs one serial warpgroup, so it is latency-bound
(measured ~6% achieved occupancy: one active warp per scheduler, no
MMA↔epilogue overlap).

This kernel INVERTS which operand persists:

- **Q resident** as the B operand `[MMA_N = N_TOKENS * num_heads, depth]`: a CTA
  owns one N_TOKENS-token block, staged once.
- **K streams** as the A operand `[BM_key = 128, depth]` through a deep SMEM
  prefetch ring, `S^T = K @ Q^T = [key, (token, head)]`, so the epilogue reduces
  over the (token, head) COLUMNS exactly like the shipped kernel (heads stay
  columns; all head counts in {4, 8, 32, 64} work uniformly, no cross-warp
  reduction).
- Grid `(batch, ceil(seq_len / N_TOKENS), num_key_parts)`: one CTA per
  (query-token block, key part). A batch-1 GLM prefill (1024 tokens,
  num_heads=32, N_TOKENS=4) is 256 CTAs on the first two axes alone, so it runs
  unsplit at `num_key_parts == 1`. Decode/MTP inverts that -- a handful of token
  blocks over a long cache -- and grid.z supplies the parallelism instead, each
  CTA streaming `_KEY_TILES_PER_CTA` tiles of its own key window. `num_key_parts`
  is a grid EXTENT, not the realized split: it is sized from the batch maximum,
  so on a ragged batch each CTA narrows it to what its own entry can feed
  (`_MIN_TILES_PER_PART`) and the surplus parts retire immediately.

Warp specialization, mirroring the MSA prefill scorer
(`Kernels/lib/msa/sparse_indexer_prefill.mojo`, PR #91938), on a 256-thread CTA at
every tile:
- WG0 (warps 0-3, threads 0-127) = score/epilogue consumer. Drains each S^T
  stage out of TMEM one row per thread (`tcgen05.ld.32x32b`, warp `w` / lane `l`
  -> row `32 * w + l`, so the 4 warps span all `BM_key` rows), applies the
  branchless relu, sums over each token's head columns entirely within the
  thread, scales by k_scale, and writes one f32 per (token, key) under the fused
  causal guard.
- WG1 (warps 4-7) = producer: warp 4 = MMA (TMEM owner + `K @ Q^T` per K tile),
  warp 5 = TMA (deep K-ring producer). Warps 6-7 are idle and register-dealloc to
  the floor; they exist only so `setmaxnreg` has a whole warpgroup to issue the
  `dec` from, which is what funds the consumer's 216. Role-to-role mbars
  (k_full/k_empty for the K ring, s_full/s_empty for the multi-stage S^T) replace
  the shipped kernel's per-iteration whole-CTA `named_barrier`; the resident Q
  owns no barrier and rides `k_full[0]`.

Scores carry no cross-key reduction (no softmax denominator), and a thread's
only global write is `output[global_token, key_local]`. Key windows are
therefore disjoint output elements: the split needs no combine pass, no
workspace, and no atomics.

Routing lives in `fp8_index_score_sm100`; see the comment there for the
measured thresholds. Two disjoint corners land here: enough token blocks to
fill the grid on their own (long prefill), or too few token blocks but a key
range deep enough that splitting it fills the grid (decode / MTP-decode). The
kernel body supports all head counts uniformly; the route currently admits
`num_heads` in {32, 64}.

NVIDIA SM100 only (SS-UMMA / TMA / tcgen05). Verified against
`nn.index_fp8.fp8_index_naive` via `test_index_fp8` and end-to-end top-k set
match via `test_mla_index_fp8`.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.sync import barrier, named_barrier
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from std.bit import next_power_of_two
from std.math import align_down, align_up, ceildiv, clamp
from std.sys import get_defined_bool, get_defined_int, size_of
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
)
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    TMATensorTile,
)

from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    TMemTile,
    elect,
    elect_mma_arrive,
    expect_bytes_pred,
    llvm_opaque_tid,
    splitk_window,
    store_global_pred,
)
from nn.attention.mha_operand import MHAOperand
from kv_cache.types import flat_scale_window, scale_align_elems
from nn.attention.gpu.nvidia.sm100.attention import SM100_RESERVED_SMEM_BYTES


# Defined locally (not imported from `sparse_index_fp8_sm100`) so that file can
# route to this one without an import cycle; they must stay type-identical to
# its aliases or the `rebind` at its call site stops compiling.
comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(MMA_N, 1, depth), _INDEX_SWIZZLE]
comptime KTMATileT[
    dtype: DType, BM_key: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(BM_key, 1, depth), _INDEX_SWIZZLE]
# The k-scale ring's descriptor: a flat `1 x KS_BOX` window on the scale pool,
# unswizzled because one scalar per key has no depth to swizzle. A SECOND
# descriptor rather than a widening of `KTMATileT` because the scales live in
# their own pool with their own paging. `KS_BOX` is one 16-byte alignment unit
# WIDER than the key tile -- see `flat_scale_window`.
comptime KSTMATileT[ks_dtype: DType, KS_BOX: Int] = TMATensorTile[
    ks_dtype, 2, Index(1, KS_BOX), Index(1, KS_BOX)
]


# WG0 (warps 0-3) = 128 epilogue/score consumers; the epilogue drains one TMEM
# lane per thread, so this must equal BM_key. The producer needs only two warps,
# one to issue MMAs and one to issue TMA loads. A second producer PAIR exists only
# to make `setmaxnreg.sync.aligned` legal: it is warpgroup-collective -- every warp
# of a warpgroup must execute it with the same operand -- so an asymmetric 216/40
# cap requires WG1 to be complete, at the price of two permanently idle warps.
#
# TWO register numbers are in play and they answer different questions.
# `setmaxnreg.inc` is the consumer REGION's allocation budget: ptxas allocates the
# code after `USETMAXREG.TRY_ALLOC` against it, and registers above the reported
# count are genuinely used there. The REPORTED count (`Used N registers`) is only
# the driver's occupancy divisor, `min(demand, 65536 / (nthreads * ctas_per_sm))`
# = 128 here. What must fit under the REPORTED count is whatever stays live across
# the role split, before the pool is redistributed -- that, not the region budget,
# is what spills.
#
# 216 + 40 = 2 * 128 returns the producers' share to the consumer exactly, which is
# what makes `TRY_ALLOC` succeed; the launcher asserts that identity at every tile.
# Deleting both `setmaxnreg` ops takes the spill 8 B -> 216 B with 32 `LDL` back in
# the tile loop.
comptime _NUM_SOFTMAX_THREADS = 128

# Hardware barrier id for the consumer warpgroup's two private syncs (the q_scale
# publish in the prologue, the TMEM drain before dealloc). Id 0 belongs to the
# whole-CTA `barrier()`, which `named_barrier` also defaults to, so a
# warpgroup-scoped sync left on the default would share it.
comptime _CONSUMER_BAR: Int32 = 1

# Producer register floor, applied to ALL FOUR of WG1's warps -- MMA, TMA and the
# two idle ones -- because `setmaxnreg.sync.aligned` is warpgroup-granular: every
# warp of a warpgroup must pass the SAME operand, so a per-warp split is UB. The
# consumer claims the rest so its TMEM->register fragment does not spill.
# Spilling is NOT monotonic in the cap, so treat this as a swept value.
# 216 + 40 = 2 * 128 returns the producers' share to the consumer exactly, which
# is what makes `TRY_ALLOC` succeed; the launcher asserts that identity.
comptime _NUM_REG_PRODUCER = 40

# S^T TMEM ring depth. 2 lets the consumer read stage `it` while the MMA writes
# `it+1`, which is what makes deferring the epilogue's WAR fence affordable.
comptime _S_TMEM_STAGES = 2

# Column chunk for the TMEM->register drain. Must stay <= 64 or ptxas runs out of
# destination registers for the `tcgen05.ld`; it must also divide MMA_N and be a
# multiple of 4. The decode twin holds a same-named constant at its own 16.
comptime _EPILOGUE_CHUNK = 32

# Columns folded per accumulator step. `SIMD[f32, W].fma` lowers to `W / 2`
# INDEPENDENT `fma.rn.f32x2` on disjoint lanes, for `W - 2` registers and NO
# extra `tcgen05.ld` -- paying only a wider per-token closing reduce.
#
# 4 is the measured optimum: on B200 at MMA_N=128, nh=32, min over reps, W=4 is
# -1.5% to -1.75% on prefill and -1.6% on decode, while W=8 is neutral (+0.1%) --
# its 12 extra `add.f32` buy latency relief the second chain already took. So 4
# is a crossover, not a point on a slope.
comptime _ACC_WIDTH = 4


# Consumer warpgroups. A tile wide enough to own the whole SM gets all of its
# The rest of the pipeline is derived from the N-tile, because TMEM decides how
# many CTAs can be co-resident.
#
# The S^T stages cost `s_stages * align_up(MMA_N, 32)` of the SM's 512 TMEM
# columns, and `tcgen05.alloc` only accepts a POWER-OF-TWO column count. So
# MMA_N=128 costs 256 (exactly half the SM, hence 2 CTAs/SM), while MMA_N=192
# uses 384 and must ALLOCATE 512 -- the whole SM, hence 1 CTA/SM.
#
# Residency is therefore a step function of the N-tile, and crossing 128 columns
# costs half the consumer warps. Measured on a batch-8 107K-key MTP step,
# MMA_N=192 cut 16.5% of the instructions and came out +2.5% in cycles -- a wash.
# Hence prefer a DIVISOR of the token count over a multiple: 3 tokens at nh=32
# (96 columns) takes the same exactness as 6 while staying on the 2-CTA/SM side.
@always_inline
def _ctas_per_sm[MMA_N: Int]() -> Int:
    return 2 if MMA_N <= 128 else 1


# Keys a slot COSTS, padded so every slot base keeps the 128-byte alignment TMA
# needs of a shared-memory destination.
@always_inline
def _ks_slot_elems[ks_dtype: DType, BM_key: Int]() -> Int:
    comptime esize = size_of[Scalar[ks_dtype]]()
    return align_up(flat_scale_window[ks_dtype, BM_key]() * esize, 128) // esize


# ===== Consumer register cap =====
# `reg_consumer` IS the budget ptxas allocates the consumer region against -- not
# a formality above the launch allocation. A one-sided dial: raise it and the
# consumer gets the room, lower it and it spills. The landscape is a VALLEY, not
# a slope -- more registers buy a more aggressive schedule that then does not fit
# -- so re-sweep rather than extrapolate, and note the cells move with any
# codegen change while the WINDOWS are the stable part.
#
# TAKING THE CAP IS NOT THE DEFAULT MOVE. Measured on B200 at the GLM-5.2 prefill
# shape (batch 1, seq 2048, cache 0, causal), min over 5 runs:
#
#   nh=32   200 (shipped)  20.74 us      184  20.59 us      232  22.56 us
#   nh=64   216 (shipped)  30.95 us      184  30.98 us      232  30.84 us
#
# 232 costs nh=32 -- every GLM-5.2 shape -- 8.8%, exactly tracking its 76 B of
# spill, while nh=64 cannot tell 216, 184 and 232 apart despite spanning 20, 0
# and 4 B. Spill is a one-way signal: it predicts a REGRESSION reliably and
# predicts nothing about a win. Do not retune a spilling cell that measures flat.


struct IndexPrefillConfig[
    dtype: DType,
    ks_dtype: DType,
    depth: Int,
    BM_key: Int,
    mma_n: Int,
    num_heads: Int,
](TrivialRegisterPassable):
    """Every derived quantity of one prefill instantiation, in one object.

    This is the `FA4Config` pattern from `nvidia/sm100/attention.mojo`, adopted
    for the same reason it exists there: the pipeline's shape, its SMEM
    accounting and its register accounting are ONE derivation, and splitting
    them across free helpers is what lets two of them disagree.

    What the split cost here, concretely: the fixed-SMEM term fed only the
    ring-depth derivation while the launcher passed a separate total, so the two
    had to be edited together by hand -- and a one-sided edit under-allocates
    the launch rather than failing it. Both are now the same accumulation:
    `smem_used == fixed_smem_bytes + k_stages * k_stage_bytes` identically.

    Residency stays OUTSIDE the struct, in `_ctas_per_sm`, because the router
    asks what residency a tile would take before it has a dtype or depth to
    build a config with.

    Parameters:
        dtype: Q/K element type.
        ks_dtype: K-scale element type.
        depth: Head dimension.
        BM_key: Keys per K tile.
        mma_n: The N tile, `n_tokens * num_heads`.
        num_heads: Index heads.
    """

    var ctas_per_sm: Int
    var nthreads: Int
    var reg_cap: Int
    var reg_consumer: Int
    var static_reg_budget: Int
    var hoist_q_scales: Bool
    var k_stages: Int
    var n_mbars: Int
    var carveout: Int
    var fixed_smem_bytes: Int
    var k_stage_bytes: Int
    var smem_used: Int

    comptime n_tokens = Self.mma_n // Self.num_heads
    # One S^T stage's TMEM columns. `tcgen05.alloc` takes a power of two, so the
    # allocation rounds up once, at the ring rather than per stage.
    comptime s_cols = align_up(Self.mma_n, 32)
    comptime tmem_cols = next_power_of_two(_S_TMEM_STAGES * Self.s_cols)

    def __init__(out self):
        self.ctas_per_sm = _ctas_per_sm[Self.mma_n]()
        # Four producer warps: two do the work (MMA, TMA) and two are idle, but
        # `setmaxnreg.sync.aligned` is warpgroup-collective, so an asymmetric
        # cap needs WG1 whole. Without it the spill goes 8 B -> 216 B with 32
        # `LDL` back in the tile loop.
        self.nthreads = _NUM_SOFTMAX_THREADS + 4 * WARP_SIZE
        # The per-thread LAUNCH allocation, which the driver divides into the
        # register file for occupancy. NOT the ceiling on the consumer region --
        # `reg_consumer` is. Must be computed BEFORE `reg_cap`, which is bounded
        # by the pool it implies. Allocation is granular in 8, so round down.
        self.static_reg_budget = align_down(
            65536 // (self.nthreads * self.ctas_per_sm), 8
        )
        # `setmaxnreg` redistributes the CTA's pool: every thread of a consumer
        # warpgroup holds the consumer cap and all four producer warps hold
        # `_NUM_REG_PRODUCER`.
        #
        # That pool is `nthreads * static_reg_budget`, NOT `65536 /
        # ctas_per_sm`. The launch allocation rounds DOWN to the 8-register
        # granule, so up to 1023 registers of the file are never handed to the
        # CTA at all. Dividing the architectural size instead returns a cap the
        # pool cannot serve, and `setmaxnreg.inc` then spins forever in
        # `USETMAXREG.TRY_ALLOC` rather than failing.
        self.reg_cap = align_down(
            (
                self.nthreads * self.static_reg_budget
                - 4 * WARP_SIZE * _NUM_REG_PRODUCER
            )
            // _NUM_SOFTMAX_THREADS,
            8,
        )
        # Keyed on the head count because the measured table forces it to be: at
        # 128 columns nh=32 needs 200 or below and nh=4 needs 208 or above, and
        # no value is clean for both. See the consumer-register note above for
        # why taking the cap is not the default move.
        if Self.mma_n == 128 and Self.num_heads == 32:
            self.reg_consumer = min(200, self.reg_cap)
        else:
            self.reg_consumer = min(
                216 if Self.mma_n <= 128 else 256, self.reg_cap
            )

        # Stage the q_scales in registers once per CTA instead of re-reading
        # them from SMEM on every key tile. A thread folds every column of its
        # own key row, so the hoist costs exactly `mma_n` registers.
        #
        # Raising the register budget does NOT raise the threshold, and that is
        # the load-bearing warning: a uniform 168 at 128 columns spills 56-220 B,
        # because with more registers ptxas commits to the aggressive schedule
        # that hoists all 8 `LDTM.x16` across the folds and then does not fit.
        # Slack is capacity, not a guarantee.
        self.hoist_q_scales = Self.mma_n <= 128

        # SMEM, accumulated ONCE. The ring depth is what the carveout still
        # affords after the fixed regions, and the total is the same
        # accumulation carried to the end -- so the launcher's
        # `shared_mem_bytes` and the depth derivation cannot disagree.
        self.carveout = (
            B200.shared_memory_per_multiprocessor // self.ctas_per_sm
            - SM100_RESERVED_SMEM_BYTES
        )
        self.fixed_smem_bytes = (
            Self.mma_n * Self.depth * size_of[Scalar[Self.dtype]]()
            + Self.mma_n * size_of[Float32]()
            + 2 * _S_TMEM_STAGES * size_of[SharedMemBarrier]()
            + size_of[UInt32]()
        )
        # One K ring slot: the tile, its per-key scales, and the
        # `k_full`/`k_empty` pair that guards it. The barriers and the scales
        # belong in the SLOT price, not in a separate fixed term -- they scale
        # with the stage count, so folding them in is what makes the division
        # exact rather than an estimate that then needs a fudge factor. The
        # scale region is what lets the consumer read `k_scale` with an `LDS`
        # instead of a two-hop dependent `LDG`; it rides the slot's own `k_full`
        # barrier, so it cannot be priced anywhere else.
        #
        # Pricing the scales in moves the derived depth 6 -> 5 at (mma_n=128,
        # 2 CTAs/SM); the other tiles are unchanged at 6 (96), 12 (192).
        self.k_stage_bytes = (
            Self.BM_key * Self.depth * size_of[Scalar[Self.dtype]]()
            + _ks_slot_elems[Self.ks_dtype, Self.BM_key]()
            * size_of[Scalar[Self.ks_dtype]]()
            + 2 * size_of[SharedMemBarrier]()
        )
        self.k_stages = (
            self.carveout - self.fixed_smem_bytes
        ) // self.k_stage_bytes
        # k_full(k_stages) + k_empty(k_stages) + s_full + s_empty. The resident
        # Q rides the first K tile's barrier.
        self.n_mbars = 2 * self.k_stages + 2 * _S_TMEM_STAGES
        self.smem_used = (
            self.fixed_smem_bytes + self.k_stages * self.k_stage_bytes
        )


# Key tiles a CTA streams before the launcher adds another grid.z part. Deep
# enough to amortize the CTA prologue (TMEM alloc, mbar init, Q staging) and to
# keep the K ring full; the launcher overrides it upward in part count when that
# many tiles per CTA would not fill a wave.
#
# 16 is the WORST-CASE choice, not the best-average one, because the two decode
# regimes want opposite counts and the launcher cannot tell them apart. What
# drives it is where the resulting grid falls inside a wave, and a uniform-depth
# batch and a ragged one present byte-identical launch arguments -- same batch,
# same `max_seq_len`, same `max_num_keys` -- while wanting opposite splits (a
# uniform batch-8 163840-key step is fastest at 18 parts and slowest at 72; a
# graded-depth one is the reverse, and 4.5x apart at the wrong end). Per-entry
# depths live in device memory, so no host-side policy can read them. 16 bounds
# the damage on the ragged side, which is what serving actually produces.
#
# TODO(cme): this leaves ~9% on the table for uniform-depth decode. Recovering
# it needs the part count chosen on the DEVICE, from per-entry depths -- a
# persistent or CLC walk -- not a different constant here.
comptime _KEY_TILES_PER_CTA = 16

# Ceiling, in waves at `_ctas_per_sm`, on the part count the amortized arm of
# the `num_key_parts` derivation may request -- see the launcher for why.
comptime _MAX_KEY_PART_WAVES = 4

# Floor on the key tiles a single grid.z part may own, applied INSIDE the kernel
# against the CTA's own per-entry tile count. The launcher can only size the part
# count from `max_num_keys` (the batch maximum), so on a ragged batch every entry
# but the deepest is over-split; this is what stops a shallow entry degenerating
# into 1-tile CTAs. Uniform batches are unaffected -- there the launcher's own
# per-part tile count already clears this floor.
comptime _MIN_TILES_PER_PART = 4


@__name(t"fp8_index_score_prefill_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(ks_tma, `nvvm.grid_constant`)
# Cap the launch register count so the config's thread count fits at its
# `ctas_per_sm` (maxntid + minctasm), mirroring MSA and FA4: without it the
# launch requests more than `65536 / nthreads` regs/thread and the warpgroup
# reg-alloc/dealloc below has no reserved pool to redistribute.
#
# Spelled as a whole config here rather than as two helper calls because these
# two numbers are one decision: `minctasm` sizes the pool that `setmaxnreg`
# redistributes, and the body's caps are derived from the same object.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(
            IndexPrefillConfig[
                dtype,
                KSOperand.dtype,
                depth,
                BM_key,
                N_TOKENS * num_heads,
                num_heads,
            ]().nthreads
        )
    )
)
# Read off the config, never spelled inline: a stale literal here is not a perf
# bug but a HANG -- `setmaxnreg.inc` spins forever when the caps it is asked for
# exceed the pool this decorator sized.
@__llvm_metadata(
    `nvvm.minctasm`=SIMDLength(
        IndexPrefillConfig[
            dtype,
            KSOperand.dtype,
            depth,
            BM_key,
            N_TOKENS * num_heads,
            num_heads,
        ]().ctas_per_sm
    )
)
def _fp8_index_score_prefill_kernel_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    ks_tma: KSTMATileT[
        KSOperand.dtype, flat_scale_window[KSOperand.dtype, BM_key]()
    ],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    output: TileTensor[.float32, OutLT, MutAnyOrigin, Engine=OutEngine],
    max_num_keys_dev: Int32,
    causal_dev: Int32,
    num_key_parts_dev: Int32,
    out_row_begin_dev: Int32,
    out_row_end_dev: Int32,
):
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    # The whole derived pipeline shape, in one object. Every stage count,
    # register cap, SMEM term and epilogue knob below reads off this -- the
    # launcher builds the identical one from the identical parameters, which is
    # what makes `shared_mem_bytes` and the body's layout the same number by
    # construction rather than by matching edits.
    comptime CFG = IndexPrefillConfig[
        dtype,
        KSOperand.dtype,
        depth,
        BM_key,
        N_TOKENS * num_heads,
        num_heads,
    ]()
    # Two producer warps are the floor: the MMA and TMA roles are separate warps
    # because each spins on its own mbarrier.
    # The N-tile is the B operand's extent, so it must be a legal UMMA N
    # (multiple of 16, <= 256 for KIND_F8F6F4 at cta_group=1). Note nothing
    # downstream checks that for us: `UMMAInsDescriptor.create` just encodes
    # `N >> 3` into the descriptor, so an illegal N would be emitted silently.
    comptime assert MMA_N % 16 == 0 and MMA_N <= 256, (
        "MMA_N (N_TOKENS * num_heads) must be a legal UMMA N -- a multiple of"
        " 16 and at most 256; got "
        + String(MMA_N)
    )
    comptime AT = DType.float32
    comptime SW = _INDEX_SWIZZLE
    comptime KS_DTYPE = KSOperand.dtype
    comptime KS_ALIGN = scale_align_elems[KS_DTYPE]()
    comptime KS_BOX = flat_scale_window[KS_DTYPE, BM_key]()
    comptime KS_SLOT = _ks_slot_elems[KS_DTYPE, BM_key]()
    comptime NSTAGE = CFG.k_stages
    comptime N_S = _S_TMEM_STAGES
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    # The `nvvm.minctasm` decorator reads the same helper; keep them in
    # lockstep -- a mismatch sizes the register pool for the wrong CTA count,
    # which hangs `setmaxnreg.inc` rather than merely costing occupancy.
    comptime assert CTAS_PER_SM == (2 if MMA_N <= 128 else 1), (
        "the body's residency must match what `nvvm.minctasm` stamped; got"
        " CTAS_PER_SM="
        + String(CTAS_PER_SM)
    )
    # ONE consumer warpgroup, which is a consequence of the tile ladder rather
    # than a choice: a second one only ever paid for a tile wide enough to own
    # the whole SM, and no such tile is reachable. The router's widest arm is
    # the alternate tile, gated on `_ctas_per_sm[MMA_N_ALT] == _ctas_per_sm[128]`
    # and so at most 128 columns. A 192-column arm existed and was MEASURED
    # SLOWER than splitting the same step across two narrow blocks (B200,
    # 2026-08-31: 1.12x on uniform decode, 1.20x once entry depths are ragged),
    # and was removed. Assert the premise rather than deriving a count from it:
    # a wider tile added later must fail to BUILD here, not silently run one
    # warpgroup on an SM-owning tile.
    comptime assert MMA_N <= 128, (
        "the consumer is single-warpgroup, which assumes 2 CTAs/SM; a tile"
        " above 128 columns takes the whole SM and would run at half the"
        " consumer warps. Restore a warpgroup count before widening. Got"
        " MMA_N="
        + String(MMA_N)
    )
    comptime CONS_WARPS = 4
    comptime CONS_THREADS = _NUM_SOFTMAX_THREADS
    # K ring depth. A PIPELINING floor, not a deadlock guard: the wait graph is
    # acyclic at any depth (the consumer only ever waits `s_full`; it arrives on
    # `k_empty` and `s_empty` and waits neither), so below the floor the TMA
    # merely spends every iteration waiting on a consumer instead of running
    # ahead of the MMA.
    #
    # One MMA per K tile, so the MMA having finished tile `t` means every issue
    # `<= t - N_S` is drained, giving a fully-started frontier of
    # `t + 1 - N_S`. A slot is released at the START of a tile (`k_empty`
    # arrives before the fold), so that frontier gates a refill and the TMA may
    # lead by `NSTAGE + 1 - N_S`. Two tiles of lead is the tight bound; the
    # `+ 2` carries one stage of margin, free at every depth this kernel derives
    # and still valid if that arrive ever moves after the fold.
    comptime assert NSTAGE >= N_S + 2, (
        "the consumer releases K slots now, so the ring must be deeper than the"
        " MMA's lag behind the fully-started frontier (N_S = "
        + String(N_S)
        + ") plus two to pipeline; got NSTAGE="
        + String(NSTAGE)
    )
    # Producer warpgroup thread roles, immediately after the consumer warps.
    comptime MMA_WARP = CONS_WARPS
    comptime TMA_WARP = MMA_WARP + 1

    # Index arithmetic runs in SIGNED 32-bit. Every quantity below is bounded by a
    # context length or a token count, and `Int` is 64-bit here, which costs a second
    # register plus an `IMAD.X`/`ISETP...EX` tail on every add and compare -- measured
    # to be what the consumer spills across the `setmaxnreg` boundary.
    #
    # SIGNED, deliberately, against the surrounding SM100 `UInt32` convention:
    # `seq_len`, `block_key_bound`, `n_key_tiles` and `n_tiles_local` are all formed
    # by subtraction and all explicitly tested `<= 0`. Unsigned would underflow those
    # into huge counts, and since `n_tiles_local` is the trip count all three warp
    # roles walk, the failure would be a HANG rather than a wrong answer. Only the
    # store OFFSET goes back to 64-bit, and it has to (see the store).
    var max_num_keys = max_num_keys_dev
    var causal = causal_dev
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)

    var start_of_seq = Int32(valid_length[b])
    var end_of_seq = Int32(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq

    # This launch owns global token rows `[out_row_begin, out_row_end)` and writes
    # them to `output` rows `[0, out_row_end - out_row_begin)`. The caller chunks
    # that window to bound the score buffer, which is the whole point of the
    # split; unchunked it is `[0, total_seq_len)` and everything below reduces to
    # the unwindowed form.
    #
    # Clamping to the window here (rather than predicating the store) is what
    # keeps the epilogue free: token blocks are indexed from `tok_lo`, so no block
    # straddles a chunk boundary, no token is scored twice, and the store's
    # liveness test just swaps `seq_len` for `tok_hi`. `seq_len` itself stays the
    # TRUE sequence length -- the causal bound is an absolute position and must
    # not see the window.
    var tok_lo = max(Int32(0), out_row_begin_dev - start_of_seq)
    var tok_hi = min(seq_len, out_row_end_dev - start_of_seq)
    # The CTA's first token block: one per CTA, so the grid steps by `N_TOKENS`
    # and `tok0` is the block base.
    var tok0 = tok_lo + Int32(block_idx.y) * Int32(N_TOKENS)
    # Folded once here so the store's row arithmetic is the same single add it
    # was before the window existed.
    var out_row0 = start_of_seq - out_row_begin_dev

    var num_keys = Int32(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len
    # `num_keys` counts tokens, for the causal bounds below. `num_rows` counts
    # what the cache holds, one pooled key per `kpool` tokens.
    var num_rows = num_keys // Int32(kpool)
    # The host bounds `max_num_keys`, but the per-entry count is device data the
    # host never sees, and on the ragged path it rests on a caller contract. So
    # check it here, in cache rows, which is what `max_num_keys` measures.
    # Keep the default assert mode. A "safe" assert in this kernel pulls in
    # `vprintf`, which inflates the emitted PTX and perturbs register
    # allocation.
    debug_assert(
        num_rows <= max_num_keys_dev,
        "fp8 index prefill: per-entry candidate rows exceed max_num_keys",
    )

    # Bail uniformly (every thread) before any collective op (TMA mbar / tcgen05
    # alloc); a divergent early return deadlocks them. A token block past the
    # sequence -- or outside this launch's row window -- produces no output (the
    # caller's -inf fill covers those rows). Uses WG0's base: if WG0 is past
    # `tok_hi` then WG1 (a later block) is too, so both WGs retire.
    #
    # A chunked launch retires most of its CTAs here, since the grid is sized by
    # the whole batch, so hoisting this above the `cache_length` load to save
    # them a global read looks free. It measured neutral (4096 tokens, 5 chunks)
    # -- the chunking overhead is not where those CTAs spend it -- so the load
    # stays where it was.
    if tok0 >= tok_hi or seq_len <= 0:
        return

    # Keys this CTA must stream: bounded by the deepest LIVE token under the
    # causal mask, which is the last token of the ONE block this CTA owns (its
    # warpgroups split that block's keys, so neither reaches past it). Each
    # token still gets its own per-key guard in the epilogue; this only trims
    # the triangle a zero-prefix fresh prefill leaves off the end.
    var last_tok = min(tok0 + Int32(N_TOKENS), tok_hi) - 1
    var block_key_bound = (
        num_keys - (seq_len - 1 - last_tok) * causal
    ) // Int32(kpool)
    # `block_key_bound` can be <= 0 (a causal bound that trims the whole block), so the
    # SUBTRACTION above stays signed. The `ceildiv` does not: `SIMD.__ceildiv__`
    # branches at COMPTIME on the dtype -- signed lowers to `-(x // -d)` with a
    # ~9-instruction correction chain, unsigned to an add and a shift. So the clamp has
    # to sit ABOVE the ceildiv; moving it below leaves the expensive form in place.
    var n_key_tiles = ceildiv(
        UInt32(max(block_key_bound, Int32(0))), UInt32(BM_key)
    )

    # grid.z splits that tile range across CTAs. Scores carry no cross-key reduction,
    # and a thread's store address is `global_token * max_num_keys + it * BM_key + row`,
    # so disjoint tile windows write disjoint elements -- no combine pass and no
    # workspace. `splitk_window` front-loads, so only trailing parts come up empty.
    #
    # The launcher sizes `num_key_parts_dev` from `max_num_keys`, a batch MAXIMUM, so on
    # a ragged batch it over-splits every entry but the deepest. Each CTA therefore
    # narrows the part count to what its OWN tile count can feed, and the surplus CTAs
    # return here, before any collective. `p_eff` depends only on CTA-uniform values, so
    # all three warp roles derive the same trip count from it -- load-bearing, because
    # windowing the consumer alone unbalances the k/s mbar handshakes and HANGS.
    var p_eff = clamp(
        ceildiv(n_key_tiles, UInt32(_MIN_TILES_PER_PART)),
        UInt32(1),
        UInt32(num_key_parts_dev),
    )
    if UInt32(block_idx.z) >= p_eff:
        return
    var win = splitk_window(
        n_key_tiles,
        p_eff,
        UInt32(block_idx.z),
    )
    var tile_begin = Int32(win[0])
    var n_tiles_local = Int32(win[1]) - tile_begin

    # Second uniform bail (an empty trailing part, or a batch entry whose causal
    # bound left it no keys). Uniform for the same reason as the one above, and
    # likewise ahead of every collective.
    if n_tiles_local <= 0:
        return

    # FA4 stateless S = K @ Q^T accumulator. `MMA_M = BM_key = 128 > 64` keeps
    # `use_ws` False (the standard, non-packed TMEM datapath). It carries no
    # handshake and no accumulator staging, so the S mbars and the stage stride
    # below are driven by this kernel. `mma_kind` has no dtype-derived default,
    # so an fp8 operand MUST select the f8f6f4 instruction family.
    comptime QK = SM100TensorAccumulator[
        dtype,
        AT,
        MMA_M=BM_key,
        MMA_N=MMA_N,
        BK=depth,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F8F6F4 if dtype.is_float8() else UMMAKind.KIND_F16,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        cta_group=1,
        num_stages=1,
    ]
    # Operand descriptor K extent, padded to the MMA_K granularity (equals the
    # accumulator's internal `padded_BK` at depth == 128).
    comptime compute_BK = align_up(depth, 16)
    comptime S_COLS = align_up(MMA_N, 32)
    comptime assert BM_key == _NUM_SOFTMAX_THREADS, (
        "the epilogue drains one TMEM lane per thread, so BM_key must equal the"
        " consumer warpgroup size"
    )
    comptime EPI_CHUNK = _EPILOGUE_CHUNK
    comptime assert MMA_N % EPI_CHUNK == 0
    comptime assert EPI_CHUNK <= 64
    # Columns folded per accumulator step; see `_ACC_WIDTH_FORCE`.
    comptime ACCW = _ACC_WIDTH
    # The fold walks columns in groups of four (one 16-byte q-scale load feeding
    # two f32x2 FFMAs), so a group must sit wholly inside one token and its base
    # must be 16-byte aligned. Both routes to this kernel supply num_heads in
    # {32, 64}, but nothing in the body enforced it.
    comptime assert EPI_CHUNK % 4 == 0 and num_heads % 4 == 0, (
        "the epilogue folds columns in groups of four, so both the epilogue"
        " chunk and num_heads must be multiples of 4 for a group never to"
        " straddle a token boundary"
    )

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=128,
        name="fp8_index_sm100_prefill_smem",
    ]()
    # Q0 resident (B operand, token block b) | [Q1 resident (token block b+1)]
    # | K ring (A operand) | k_scale ring | q_scale Q0 | [q_scale Q1] | mbars
    # | tmem ptr.
    # Q0 rides the first K tile's `k_full` barrier (`issue_k[with_q=True]`);
    # Q1 lands on its own dedicated `q1` mbar (FA4's `q1_wait_mbar` pattern) so
    # its arrival stays off the K-ring barriers.
    #
    # The k_scale ring is slot-for-slot the K ring's: slot `s` holds the scales
    # of the key rows the MMA reads from `k_smem` slot `s`, landed by a second
    # TMA on that slot's own `k_full`. It sits immediately after the K ring so
    # its base inherits the 128-B alignment TMA needs, and `KS_SLOT` is padded
    # to keep every later slot on that alignment. A slot stages `KS_BOX` scales,
    # one alignment unit more than the tile's `BM_key`, because the window can
    # only start on a 16-byte boundary (`_ks_align_elems`); the consumer skips
    # the leading `ks_off` residual.
    #
    # ONE resident Q region and one q_scale region, shared by every consumer
    # warpgroup.
    var q_smem = smem
    var k_smem = smem + q_elems
    var ks_smem = (k_smem + NSTAGE * k_elems).bitcast[Scalar[KS_DTYPE]]()
    var qs_smem = (ks_smem + NSTAGE * KS_SLOT).bitcast[Float32]()
    var mbar = (qs_smem + MMA_N).bitcast[SharedMemBarrier]()
    # Offsets are named because the lane-parallel init below maps a thread index
    # to an arrival count, which makes them part of the layout contract rather
    # than a one-off pointer bump. `s_empty` last is load-bearing -- see the init.
    #
    # ONE S^T ring of `N_S` slots at every warpgroup count: the single MMA warp
    # issues one MMA per K tile into consecutive slots, so K tile `t` lands on
    # slot `t mod N_S` and warpgroup `g` walks the ring from seed `g`.
    comptime K_FULL_OFF = 0
    comptime K_EMPTY_OFF = K_FULL_OFF + NSTAGE
    comptime S_FULL_OFF = K_EMPTY_OFF + NSTAGE
    comptime S_EMPTY_OFF = S_FULL_OFF + N_S
    comptime N_MBAR = S_EMPTY_OFF + N_S
    comptime assert N_MBAR == CFG.n_mbars
    var k_full = mbar + K_FULL_OFF
    var k_empty = mbar + K_EMPTY_OFF
    var s_full = mbar + S_FULL_OFF
    var s_empty = mbar + S_EMPTY_OFF
    var ptr_tmem = (mbar + N_MBAR).bitcast[UInt32]()

    comptime q_flat_layout = tt_row_major[q_elems]()
    comptime k_flat_layout = tt_row_major[k_elems]()
    comptime ks_flat_layout = tt_row_major[KS_BOX]()

    # Threads that must release a K slot before the TMA may refill it: the MMA
    # warp (one elected arrive) plus every consumer thread that reads a
    # `k_scale` out of that slot. A tile belongs to exactly one warpgroup, so
    # exactly one warpgroup's 128 threads touch a given slot on a given lap.
    # Each thread arrives for itself because `mbarrier.arrive` carries the
    # release that publishes its own `LDS` -- a batched arrive would not.
    comptime KS_READERS = _NUM_SOFTMAX_THREADS
    comptime K_EMPTY_ARRIVES = 1 + KS_READERS

    # Lane-parallel init, mirroring FA4's `FA4MiscMBars.init`: barrier `i` is
    # initialized by thread `i`, so all N_MBAR `mbarrier.init` issue as one `STS.64`
    # with a lane-indexed address instead of N_MBAR serialized stores from thread 0.
    #
    # Two classes carry a non-default count: `k_empty` (the MMA plus the
    # consumers that read the slot's scales) and `s_empty` (one arrival per
    # consumer thread). `s_full` is armed by a single tcgen05 commit and every
    # remaining producer barrier by one TMA completion, so those take 1.
    # `k_empty` is one contiguous range at the FRONT and `s_empty` one at the
    # BACK, so the count map stays a `SEL` pair rather than a branch chain --
    # keep both blocks contiguous or this grows one. Indexed by `tid` rather than
    # lane so it stays correct if N_MBAR ever passes WARP_SIZE.
    comptime assert N_MBAR <= CFG.nthreads
    if tid < N_MBAR:
        mbar[tid].init(
            Int32(K_EMPTY_ARRIVES) if (
                tid >= K_EMPTY_OFF and tid < S_FULL_OFF
            ) else Int32(_NUM_SOFTMAX_THREADS) if (
                tid >= S_EMPTY_OFF
            ) else Int32(
                1
            )
        )

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp (the MMA warp).
    # Release the lock right after so co-resident CTAs can allocate. `tcgen05.alloc`
    # takes a POWER-OF-TWO column count, so the stages' footprint is rounded up for the
    # allocation (MMA_N=192 uses 384 and allocates 512) while the stage stride stays
    # `S_COLS` -- the waste is dead columns at the top, not a gap between stages.
    #
    # ONE ring of `N_S` slots, `S_COLS` columns each. The forced arm takes
    # `N_S = 4`, so at `MMA_N=128` that is 512 columns -- the whole SM, which is
    # why the arm is scoped to `MMA_N <= 128`. Residency is NOT decided here:
    # `_ctas_per_sm` decides it and `nvvm.minctasm` stamps it, and this assert
    # only catches the two disagreeing.
    comptime TMEM_USED_COLS = N_S * S_COLS
    comptime TMEM_COLS = UInt32(next_power_of_two(TMEM_USED_COLS))
    comptime assert Int(TMEM_COLS) * CTAS_PER_SM <= 512, (
        "S^T stages need "
        + String(TMEM_COLS)
        + " TMEM columns (rounded up to a power of two from "
        + String(TMEM_USED_COLS)
        + "), which exceeds the SM's 512 at CTAS_PER_SM="
        + String(CTAS_PER_SM)
    )
    var wid = warp_id[broadcast=True]()
    barrier()

    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()
    comptime ks_bytes = KS_BOX * size_of[KS_DTYPE]()

    if wid < CONS_WARPS:
        warpgroup_reg_alloc[CFG.reg_consumer]()

        # q_scale staging, one f32 per (token, head) column. Both warpgroups
        # fold the same block against the same Q -- the key-tile split, not the
        # Q, distinguishes them -- so there is one region, shared. This sits INSIDE the
        # consumer branch rather than the whole-CTA prologue because it is a dependent
        # global load and no producer warp reads the result, which is what let the
        # prologue's second whole-CTA `barrier()` go: the TMA warp now reaches its first
        # K issue without waiting on 128 cold global loads. A `bar.sync` fences
        # intra-CTA `st.shared`/`ld.shared`, so no `fence_async_view_proxy` is needed.
        #
        # One wave per `STAGE_THREADS` columns, with the bound check emitted ONLY
        # for a wave that does not fill. Waves rather than `tid < MMA_N` because
        # the staging must not reach past the consumer warpgroups: at MMA_N=192
        # the old form wrote from warps 4 and 5 too.
        #
        comptime STAGE_THREADS = CONS_THREADS
        var stage_tid = tid

        @__parameter
        @always_inline
        def stage_qs(col: Int):
            var qs_tok = Int32(col // num_heads)
            if tok0 + qs_tok < seq_len:
                qs_smem[col] = q_s[
                    start_of_seq + tok0 + qs_tok, col % num_heads
                ][0]
            else:
                qs_smem[col] = 0.0

        comptime for w in range(ceildiv(MMA_N, STAGE_THREADS)):
            comptime col_base = w * STAGE_THREADS
            comptime if col_base + STAGE_THREADS > MMA_N:
                if stage_tid + col_base < MMA_N:
                    stage_qs(stage_tid + col_base)
            else:
                stage_qs(stage_tid + col_base)
        named_barrier[Int32(CONS_THREADS + WARP_SIZE)](_CONSUMER_BAR)
        var tmem_addr: UInt32 = ptr_tmem[0]
        # `tcgen05_ld[datapaths=32]` picks its TMEM sub-partition WARPGROUP-relative
        # (`warp_id % 4`), mapping warp w, lane l -> accumulator row
        # `WARP_SIZE * (w % 4) + l`. So each consumer warpgroup's 4 warps cover all
        # BM_key rows one-per-thread, and several warpgroups each cover the same rows
        # of their OWN stage. A thread owning a whole key row makes the head sum
        # thread-local (no cross-lane reduction) and puts 32 consecutive `key_local`
        # in each warp, so each store is one 128B transaction. Masking `tid` is
        # therefore the right spelling at any warpgroup count -- it is already
        # warpgroup-local -- and gives `(wid % 4) * 32 + lane_id()` identically while
        # reading `tid` once, avoiding a second `S2R`.
        # `llvm_opaque_tid` is FA4's anti-hoist intrinsic; it does NOT bind here
        # (exactly one `%tid.x` read either way, in the prologue ahead of
        # `TRY_ALLOC`, because the q-scale staging above already needs `tid`), and is
        # kept only for the cheaper spelling.
        var row = Int32(llvm_opaque_tid() & (_NUM_SOFTMAX_THREADS - 1))
        # `PipelineState[N].step()` is `index += 1; if index == N { index = 0;
        # phase ^= 1 }`. The consumer walks the S^T ring in issue order, so
        # after `k` tiles it sits on the slot and lap parity of issue `k` --
        # what the MMA wrote.
        var c_state = PipelineState[N_S]()

        # Which K ring slot holds this warpgroup's current tile. The MMA warp
        # walks tiles 0,1,2,... assigning slots 0,1,2,... mod NSTAGE, so local
        # tile `j` is always in slot `j % NSTAGE` -- and this cursor is built and
        # stepped exactly like `c_state` so it lands on the same tile's slot at
        # every iteration. Only the INDEX is read (the slot's data is already
        # published by `s_full`), so the phase is unused here.
        var k_state = PipelineState[NSTAGE]()

        # How many keys the producer's alignment round-down put in front of this
        # CTA's first key, and so where a slot's tile actually starts. Read once,
        # not per tile: the per-tile term the producer adds (`it * BM_key`) is a
        # multiple of `KS_ALIGN`, so the residual is the batch entry's own and is
        # the same on every tile. It is zero on every paged operand (`page_size %
        # BM_key == 0` forces it) and non-zero only on a ragged batch whose
        # per-entry key count is not a multiple of `KS_ALIGN`.
        var ks_off = Int32(
            ks_operand.row_idx(UInt32(b), UInt32(tile_begin * Int32(BM_key)))
            & UInt32(KS_ALIGN - 1)
        )

        # The q_scales a thread needs are CTA-invariant, yet the tile loop below
        # re-read all of them on EVERY key tile: measured at 128 columns as 1,295,264
        # `LDS`, 7.8% of the instruction stream and 36% of the shared-load pipe, with
        # the stalls landing on the consuming `FFMA2` rather than on the `LDS` itself.
        #
        # `qs_reg` is the single access path for the fold either way; only the FILL
        # SITE is conditional. Where the working set fits (see `hoist_q_scales`) it is
        # filled once here, taking the in-loop `LDS` count to ZERO; otherwise each
        # chunk fills its own four in place below. Read FOUR at a time: adjacent scalar
        # reads coalesce into one 16-byte access and an explicit width-2 load does not
        # re-merge. Every index MUST stay comptime -- a runtime index forces the array
        # to local memory. Each warpgroup loads its OWN q-scales (Q0's or Q1's).
        var qs_reg = Array[Scalar[AT], MMA_N](uninitialized=True)

        # The epilogue stores in place today because a token's sum is final at
        # its last column, which collapses the accumulator to one f32x2. That
        # collapse is NOT what this undoes -- the fold still collapses per token
        # and `acc` still resets. What is deferred is only the STORE, so the
        # cost is `N_TOKENS` finished f32 values rather than the `N_TOKENS`
        # running accumulators plus all `MMA_N` q-scales (~250 registers) that
        # the original design rejected.
        #
        # Every index MUST stay comptime, exactly as for `qs_reg`: a runtime
        # index forces the array to local memory and the arm becomes a spill
        # experiment instead of a scheduling one.
        comptime if CFG.hoist_q_scales:
            comptime for j in range(MMA_N // 4):
                var qs4 = qs_smem.unsafe_load[width=4, alignment=16](4 * j)
                comptime for e in range(4):
                    qs_reg[4 * j + e] = qs4[e]

        # `k_scale` now comes out of SMEM, staged by the TMA warp into this
        # tile's own K ring slot. What it replaces was a TWO-HOP dependent global
        # load (block table -> scale row) issued by every consumer thread and
        # rotated one key tile ahead to cover its own miss: 37% L2 hit, with 5.5%
        # of warp-stall samples landing on the `FMUL` that consumed it against
        # 0.2% on the identical one a token later. An `LDS` has none of that
        # variance, the producer warp that now issues the load is 57-70% idle,
        # and under distinct-Q it also stops both warpgroups gathering the SAME
        # 128 scales independently.
        #
        # A row past `num_rows` reads pool data rather than the `0.0` the old
        # guard returned. Safe for the same reason the K tile's own out-of-range
        # rows are: the epilogue store is gated on `key_local < key_bound`, and
        # `key_bound <= num_rows` always, so such a score is folded and then
        # discarded, never written.
        # Raw base of the score buffer.
        var out_base = output.ptr

        # `n_tiles_local` is `Int32`, so `range` yields an `Int32` induction
        # variable directly (`range.mojo:495`) -- the whole key-index chain
        # stays 32-bit with no per-iteration cast, which is what the narrowing
        # rule requires.
        for tile_i in range(n_tiles_local):
            var key_local = (tile_begin + tile_i) * Int32(BM_key) + row

            # `PipelineState.index()` is already `UInt32`; keep it that way
            # rather than round-tripping through 64-bit `Int`.
            var cs = c_state.index()
            s_full[cs].wait(c_state.phase())
            var s_it = tmem_addr + cs * UInt32(S_COLS)

            # No `k_full` wait here: the MMA warp waited it before issuing, and
            # `s_full` is armed by that MMA's completion, so the scales are
            # already visible transitively. Read then release immediately --
            # before the fold, not after -- so the TMA gets the slot back as
            # early as possible now that the consumer sits in its WAR loop.
            # `mbarrier.arrive`'s CTA-scope release pattern orders the `LDS`
            # ahead of the arrive, the same way it does for the MMA's own
            # `k_empty` arrive.
            var ks = k_state.index()
            var k_scale = ks_smem[
                ks * UInt32(KS_SLOT) + UInt32(ks_off + row)
            ].cast[.float32]()
            _ = k_empty[ks].arrive()

            # A token owns a CONTIGUOUS column range (`col // num_heads`), so
            # its sum is final at its last column: store it right there and the
            # accumulator collapses to one `SIMD[AT, ACCW]`, rather than keeping
            # every token sum and all `MMA_N` q-scales live at once (~250
            # registers, which spilled).
            #
            # `num_heads` and `EPI_CHUNK` are both powers of two, so a chunk
            # never straddles a token partway and the completion test stays
            # comptime. A *runtime* token index would spill `acc` to local
            # memory.
            @__parameter
            @always_inline
            def consume_group[
                col: Int
            ](frag: Array[Scalar[AT], EPI_CHUNK], mut acc: SIMD[AT, ACCW],):
                # Read FOUR scales at a time: adjacent scalar reads coalesce
                # into one 16-byte access and an explicit width-2 load does not
                # re-merge. Every index MUST stay comptime -- a runtime index
                # forces `qs_reg` to local memory.
                comptime if not CFG.hoist_q_scales:
                    comptime for q in range(ACCW // 4):
                        var qsg = qs_smem.unsafe_load[width=4, alignment=16](
                            col + 4 * q
                        )
                        comptime for e in range(4):
                            qs_reg[col + 4 * q + e] = qsg[e]
                # `ACCW / 2` disjoint lane pairs, so this one `.fma` is `ACCW / 2`
                # independent `fma.rn.f32x2` -- the chains differ, the
                # instruction count does not. `.fma()` is required over
                # `a * b + c`: LLVM does not contract an f32 pair into one
                # FFMA2. The relu is a vector op but PTX has no `max.f32x2` at
                # any ISA version, so it lowers to one FMNMX per column.
                var raw = SIMD[AT, ACCW]()
                var qsv = SIMD[AT, ACCW]()
                comptime for e in range(ACCW):
                    raw[e] = frag[col % EPI_CHUNK + e]
                    qsv[e] = qs_reg[col + e]
                acc = max(raw, SIMD[AT, ACCW](0)).fma(qsv, acc)
                comptime if (col + ACCW) % num_heads == 0:
                    comptime t = col // num_heads
                    var tok_local = tok0 + Int32(t)
                    # Fused causal mask (branchless): token tok_local sees
                    # keys up to cache_len + tok_local, so forbidden slots
                    # are left for the caller (the MLA caller pre-fills
                    # -Float32.MAX) and the separate mask pass is skipped.
                    # The liveness half is memory safety rather than
                    # masking -- a dead token's row belongs to the NEXT
                    # batch entry.
                    #
                    # The guard skips no work worth skipping (0.03%
                    # divergence), but as a BRANCH it costs a
                    # `BSSY`/`BRA`/`NOP`/`BSYNC` quartet per token per key
                    # tile: 4.6% of the instruction stream at 96 columns,
                    # 10.6% at 128. store_global_pred folds it into a PTX
                    # `@%p st.global.b32` instead, measured at -5.4% to
                    # 0.0% wall clock over five shapes with none
                    # regressing. Predicating only the STORE leaves every
                    # register dependency intact, so ptxas does NOT hoist
                    # the 8 `LDTM.x16` across the folds: zero `STL`/`LDL`
                    # in both arms at MMA_N 96 and 128, nh 32 and 64.
                    var key_bound = (
                        num_keys - (seq_len - 1 - tok_local) * causal
                    ) // Int32(kpool)
                    # The OFFSET must be 64-bit: the score buffer is
                    # `total_seq_len * max_num_keys` f32, so the flat index
                    # passes 2^31 at ~13.1K tokens x 163840 keys and 2^32
                    # at 32K x 163840, both reachable by configuration with
                    # no allocation guard on that path. Both factors are
                    # `Int32` though, so this is one `IMAD.WIDE` (32x32->64)
                    # rather than a 64-bit multiply. Do NOT narrow it.
                    var out_row = out_row0 + tok_local
                    store_global_pred(
                        out_base
                        + (Int(out_row) * Int(max_num_keys) + Int(key_local)),
                        k_scale * acc.reduce_add(),
                        Int32(key_local < key_bound and tok_local < tok_hi),
                    )
                    acc = SIMD[AT, ACCW](0)

            # Drain this thread's key row in `EPI_CHUNK`-column chunks. The
            # chunk loads carry no wait between them, so they pipeline against
            # the folds: `tcgen05.ld` register outputs are automatically
            # ordered, and the single `tcgen05_load_wait` is only the WAR fence
            # before the stage is released.
            var acc = SIMD[AT, ACCW](0)
            comptime for c in range(MMA_N // EPI_CHUNK):
                var frag = TMemTile[AT, BM_key, EPI_CHUNK](
                    s_it + UInt32(c * EPI_CHUNK)
                ).load_async()
                comptime for g in range(EPI_CHUNK // ACCW):
                    consume_group[c * EPI_CHUNK + ACCW * g](frag, acc)
            tcgen05_load_wait()
            tcgen05_fence_before()
            _ = s_empty[cs].arrive()

            c_state.step()
            k_state.step()

        # Drain across the consumer warps: every consumer's last `s_empty`
        # arrive happens-before this barrier, so no S^T stage is live when the
        # MMA warp frees TMEM.
        named_barrier[Int32(CONS_THREADS)](_CONSUMER_BAR)
        if wid == 0:
            tcgen05_dealloc[1](tmem_addr, TMEM_COLS)
    else:
        if wid == MMA_WARP:
            tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
            tcgen05_release_allocation_lock[1]()
            # The role runs warp-collectively and elects one lane per issue:
            # `SM100TensorAccumulator.mma` broadcasts the accumulator's TMEM
            # address from lane 0 (`shfl.sync` over the full warp mask), so
            # calling it from a single-lane region hangs the warp on a
            # convergence barrier the other 31 lanes never reach.
            warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            named_barrier[Int32(CONS_THREADS + WARP_SIZE)](_CONSUMER_BAR)
            var tmem_addr: UInt32 = ptr_tmem[0]
            var e = elect()
            var kc_state = PipelineState[NSTAGE]()
            # The MMA warp is the PRODUCER of the S^T stages, so its WAR state starts
            # pre-flipped: `wait(1)` on a fresh mbar tests a phase that has not been
            # reached and falls through, replacing a prologue that pre-armed `s_empty`
            # with 256 arrivals purely to make that same first wait pass. Same device
            # as the K ring's `kp_state` below.
            var sp_state = PipelineState[N_S](0, 1, 0)
            # No separate wait for the resident Q0: it lands on `k_full[0]` with the
            # first K tile, so iteration 0's `k_full` wait below covers it. `q_smem`
            # is written once and never recycled, so later iterations reading it
            # behind a re-armed `k_full[0]` is not a hazard.
            #
            # Trip count must match the consumer's and the load warp's exactly: the
            # k/s mbar handshakes and both `PipelineState` phases stay in lockstep
            # only because all three roles walk the same tile window.
            var q0 = smem_descriptor[
                BMN=MMA_N,
                BK=compute_BK,
                swizzle_mode=SW,
                is_k_major=True,
            ](q_smem)
            var k = smem_descriptor[
                BMN=BM_key,
                BK=compute_BK,
                swizzle_mode=SW,
                is_k_major=True,
            ](k_smem)
            for _ in range(n_tiles_local):
                var s = kc_state.index()
                k_full[s].wait(kc_state.phase())
                # One MMA per K tile into the shared ring, so the ring's issue
                # counter IS the tile counter. `sp_state` tracks it and is
                # re-read and stepped here rather than hoisted.
                var st = sp_state.index()
                # WAR: the consumers must have drained this S stage before the
                # MMA overwrites it. The first pass over each stage falls through
                # on the pre-flipped phase (nothing to drain yet); after that
                # this is the consumer's `release_stage()`.
                s_empty[Int(st)].wait(sp_state.phase())
                QK.mma(
                    k + s * UInt32(k_elems),
                    q0,
                    tmem_addr + st * UInt32(S_COLS),
                    c_scale=0,
                    elect=e,
                )
                elect_mma_arrive(s_full + Int(st), e)
                sp_state.step()
                # Release K stage s only after the MMA has drained it
                # (tcgen05.commit tracks the async MMA); a plain mbar arrive
                # would let the load warp overwrite K mid-read.
                elect_mma_arrive(k_empty + s, e)
                kc_state.step()
        elif wid == TMA_WARP:
            warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()
            var kp_state = PipelineState[NSTAGE](0, 1, 0)
            var n_prefetch = min(Int32(NSTAGE), n_tiles_local)
            var e = elect()

            # `with_q` carries the resident Q on this tile's barrier as well, and holds
            # for the PEELED FIRST ISSUE ONLY (see the prologue below): Q is staged
            # once, so every later arming of a stage -- including stage 0's refill --
            # expects `k_bytes` alone.
            #
            # One expect-bytes with the bytes SUMMED is the only legal shape, not one
            # call per copy: it lowers to `mbarrier.arrive.expect_tx`, so it IS the
            # producer's single arrival, and against an init count of 1 a second call
            # would drive the pending-arrival count below zero -- out of spec, not
            # merely redundant. Summed, both TMAs deposit into the one tx counter and
            # the barrier fires when the last byte of either lands.
            #
            # The single-lane guard rides an `@%p` on each instruction rather than an
            # `if e != 0:`. That is NOT a branch removal -- ptxas already if-converts
            # the guarded form and both shapes emit the same SASS at the issue. It is
            # worth 8 fewer instructions of uniform-datapath setup per sidecar, and
            # consistency with every other SM100 producer.
            @__parameter
            @always_inline
            def issue_k[
                with_q: Bool = False
            ](it: Int32, state: PipelineState[NSTAGE]):
                var s = state.index()
                # `k_row0` stays `Int`: `async_copy_3d` takes
                # `coords: Tuple[Int, Int, Int]`, so narrowing it only adds a
                # widening cast back at the call.
                var k_row0 = Int(
                    k_operand.row_idx(UInt32(b), UInt32(it * Int32(BM_key)))
                )
                # The scales' own row index, NOT `k_row0`: a paged scale pool
                # resolves its block through `scales_lookup_table` and strides
                # by its own block pitch, so reusing the K row silently reads
                # another block whenever the two LUTs differ.
                #
                # Rounded DOWN to `KS_ALIGN`: a scale "row" is a single scalar,
                # so this index IS the innermost TMA coordinate and must land on
                # a 16-byte boundary, which a ragged batch's per-entry base does
                # not. The window is `KS_ALIGN` keys wider to cover the shift and
                # the consumer skips the residual. (`k_row0` needs no such fix --
                # a K row is a whole `depth`-wide key, so it is always aligned.)
                var ks_row0 = Int(
                    ks_operand.row_idx(UInt32(b), UInt32(it * Int32(BM_key)))
                ) & ~(KS_ALIGN - 1)
                var k_dst = TileTensor[
                    dtype, type_of(k_flat_layout), address_space=.SHARED
                ](k_smem + s * UInt32(k_elems), k_flat_layout)
                var ks_dst = TileTensor[
                    KS_DTYPE,
                    type_of(ks_flat_layout),
                    address_space=.SHARED,
                ](ks_smem + s * UInt32(KS_SLOT), ks_flat_layout)
                expect_bytes_pred(
                    k_full + s,
                    Int32(k_bytes + ks_bytes + q_bytes) if with_q else Int32(
                        k_bytes + ks_bytes
                    ),
                    e,
                )
                k_tma.async_copy_3d_elect(k_dst, k_full[s], (0, 0, k_row0), e)
                # Rides the K tile's barrier for the same reason Q0 does: the
                # bytes are summed into the one `expect_tx`, so the slot opens
                # when the last byte of either copy lands and the consumer needs
                # no wait of its own -- `s_full` already happens-after this.
                ks_tma.async_copy_elect(ks_dst, k_full[s], (ks_row0, 0), e)
                comptime if with_q:
                    var q_dst = TileTensor[
                        dtype, type_of(q_flat_layout), address_space=.SHARED
                    ](q_smem, q_flat_layout)
                    q_tma.async_copy_3d_elect(
                        q_dst,
                        k_full[s],
                        (0, 0, Int(start_of_seq + tok0) * num_heads),
                        e,
                    )

            # Prologue: fill the first NSTAGE stages (fresh, no k_empty wait).
            # `tile_begin` shifts only the tile ADDRESS -- the ring index/phase
            # sequence is a function of the count, so it is unshifted.
            #
            # Iteration 0 is peeled to carry the resident Q0. Always taken:
            # `n_tiles_local >= 1` is guaranteed by the bail above, so
            # `n_prefetch >= 1`. Peeled rather than an `i == 0` test because `i` is
            # a runtime value and `with_q` must be comptime.
            issue_k[with_q=True](tile_begin, kp_state)
            kp_state.step()
            for i in range(Int32(1), n_prefetch):
                issue_k(tile_begin + i, kp_state)
                kp_state.step()
            # Refills: wait k_empty at that stage/phase (MMA done with the prior
            # occupant, tile i-NSTAGE) before reissuing.
            for i in range(n_prefetch, n_tiles_local):
                k_empty[kp_state.index()].wait(kp_state.phase())
                issue_k(tile_begin + i, kp_state)
                kp_state.step()
        else:
            # Idle warps 6-7, and the ONLY reason they are launched: they must
            # dealloc the SAME count as warps 4-5, because
            # `setmaxnreg.sync.aligned` is warpgroup-collective and a differing
            # operand within WG1 is UB. Drop `setmaxnreg` and these warps have no
            # purpose at all.
            warpgroup_reg_dealloc[_NUM_REG_PRODUCER]()


# Route a shape here when a sequence spans at least this many token blocks: the
# token-block grid alone then fills the machine without a key split.
comptime _PREFILL_MIN_TOKEN_TILES = 16

# The other direction: too FEW token blocks to fill the grid, but a key range
# deep enough that splitting it does. This is decode / MTP-decode against a long
# cache, where the K-resident scorer degenerates to one CTA per key tile doing a
# single MMA -- it pays a full CTA prologue (TMEM alloc, mbar init, Q staging,
# k_scale gather) per 128 keys and never pipelines the K stream. Here one CTA
# instead streams `_KEY_TILES_PER_CTA` tiles through the ring behind a resident
# Q tile.
comptime _KEYSPLIT_MAX_TOKEN_TILES = 4
comptime _KEYSPLIT_MIN_KEY_TILES = 64
# Load-bearing beyond this route: the alternate N-tile's block-count clause reads
# `max(token_tiles, _KEYSPLIT_MAX_TOKEN_TILES)`, so at K=4 with N_ALT=3 only a
# `max_seq_len` in {3, 6, 9} can reach the narrower tile -- speculative widths only,
# which is the property every `MMA_N < 128` claim in this file is scoped on. Raising K
# widens that toward genuine prefill, so re-tuning it must re-check the alternate
# tile, not just the key split.


@always_inline
def _is_keysplit_shape[
    num_heads: Int, BM_key: Int
](max_seq_len: Int, max_num_keys: Int) -> Bool:
    """Whether a shape is decode-shaped: few token blocks, deep key range.

    One definition, read by both the router (which admits such a shape to the
    Q-resident kernel) and the launcher (which relaxes its key-split guard for
    it). Deliberately measured against the DEFAULT `128 // num_heads` tile
    rather than the tile actually launched: a wider tile makes any shape look
    like fewer token blocks, and a 17-token prefill batch on the 8-token tile
    would otherwise read as decode.

    Parameters:
        num_heads: Query index heads.
        BM_key: Keys per tile.

    Args:
        max_seq_len: Batch maximum of new query tokens.
        max_num_keys: Row stride of the score buffer; an upper bound on any
            entry's key count, and under graph capture a frozen one.

    Returns:
        True when the token-block grid alone cannot fill the machine but the
        key range is deep enough that splitting it can.
    """
    return (
        ceildiv(max_seq_len, 128 // num_heads) <= _KEYSPLIT_MAX_TOKEN_TILES
        and ceildiv(max_num_keys, BM_key) >= _KEYSPLIT_MIN_KEY_TILES
    )


@always_inline
def fp8_index_score_sm100_prefill[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[mut=False, .uint32, ..., Engine=VLEngine],
    q_s: TileTensor[mut=False, .float32, ..., Engine=QSEngine],
    output: TileTensor[.float32, ..., Engine=OutEngine],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Int,
    ctx: DeviceContext,
    out_row_begin: Int = 0,
) raises:
    """Enqueue the warp-specialized K-streaming prefill scorer into `output`.

    One CTA per (batch, token block, key part). A part streams its own window of
    the token block's causally-reachable keys; the windows are disjoint and carry
    no reduction, so there is no combine pass. `num_key_parts` bounds the split
    rather than fixing it -- a batch entry too shallow to feed that many parts
    narrows it in-kernel and the surplus CTAs retire before any collective.
    Called from `fp8_index_score_sm100` for both the many-token-block prefill
    route and the key-split decode/MTP route.

    `output` holds the global token rows `[out_row_begin, out_row_begin +
    output.dim[0]())`, so a caller that cannot afford the whole
    `total_seq_len x max_num_keys` score matrix can fill it a row-window at a
    time. The default covers every row and is the unwindowed launch.
    """
    # The window bounds the token blocks any entry can contribute, so a chunked
    # launch does not pay for a grid sized to the whole batch.
    var out_rows = Int(output.dim[0]())
    comptime sm_count = ctx.default_device_info.sm_count
    comptime MMA_N = N_TOKENS * num_heads
    comptime CTAS_PER_SM = _ctas_per_sm[MMA_N]()
    # One token block per CTA, so grid.y is one CTA per token block.
    var token_blocks = ceildiv(min(max_seq_len, out_rows), N_TOKENS)
    # Split the key range over grid.z only when the (batch, token block) grid alone
    # leaves SMs idle -- splitting an already-full grid costs pipeline depth for
    # nothing (the K-resident scorer measured -29% doing exactly that). Target
    # `_KEY_TILES_PER_CTA` tiles per CTA, but take more parts if that is what it costs
    # to reach a wave at `_ctas_per_sm`.
    #
    # The wave-fill arm FLOORS: it wants the largest part count whose grid still fits
    # one wave, and `ceildiv` overshoots by construction (at base_ctas=64 it gave
    # 5 -> 320 CTAs = 2 waves where 4 -> 256 = 1 wave delivers the same tiles per CTA).
    #
    # The amortized arm is CAPPED at `_MAX_KEY_PART_WAVES` waves: `max_num_keys` is
    # METADATA, and a captured decode graph freezes it at its capture-time bound (1M
    # for a full-context GLM graph) while the live keys stay orders of magnitude
    # smaller, so a part count proportional to `key_tiles` turns that gap into empty
    # CTAs. Past a few waves the extra parts buy no parallelism even when the
    # bound matches the runtime key range -- capped parts just stream more tiles each, which the
    # K-ring amortizes better than more prologues would.
    #
    # `base_ctas` counts the CTAs that will WORK, not the ones the grid totals.
    # `batch_size * ceildiv(max_seq_len, N_TOKENS)` is the total, and on a ragged
    # batch nearly all of the excess retires at the
    # `block_idx.y * N_TOKENS >= seq_len` bail before doing anything. A batch that
    # launches past `sm_count` while working well under it would otherwise be read
    # as a full machine, decline the split, and leave the deepest entry streaming
    # its whole key range on ONE CTA. The token total is exact (`output` is
    # `[total_seq, max_num_keys]` by both entries' contract) and the `min` makes
    # this a pure RELAXATION: `base_ctas` can only fall, so every uniform batch
    # keeps its grid byte for byte and no shape loses a split it already had.
    #
    # The second clause lets a DECODE-shaped launch split even when the token-block
    # grid already fills the machine. Without it, at batch >= sm_count no key part
    # is ever assigned, so a ragged CONTEXT mix -- uniform query width, wildly
    # different cache depths -- runs the deepest entry on one CTA beside SMs that
    # finished early, and the host cannot see that skew because per-entry depths
    # are device data. Measured worth 2.3x-4.5x at batch >= 148. It is restricted
    # to `_is_keysplit_shape` because relaxing it at prefill widths would multiply
    # an already-full grid for no extra parallelism.
    # Each entry occupies `ceildiv(seq_len, N_TOKENS)` blocks, and summing that
    # over the batch without per-entry lengths is exactly the second term, which
    # over-counts by at most `batch_size - 1`. The `max(1, ...)` guards an empty
    # batch only; `wave_parts` below divides by this. The first term is bounded
    # by the row window (`out_rows`), so a chunked score-matrix launch sizes
    # its grid to the rows it fills rather than the whole batch (see #96716);
    # an unwindowed caller has `out_rows >= max_seq_len` and the `min` is a no-op.
    var base_ctas = max(
        1,
        min(
            batch_size * ceildiv(min(max_seq_len, out_rows), N_TOKENS),
            (Int(output.dim[0]()) + batch_size * (N_TOKENS - 1)) // N_TOKENS,
        ),
    )
    var key_tiles = ceildiv(max_num_keys, BM_key)
    var num_key_parts = 1
    if base_ctas < sm_count or _is_keysplit_shape[num_heads, BM_key](
        max_seq_len, max_num_keys
    ):
        var wave_parts = (CTAS_PER_SM * sm_count) // base_ctas
        num_key_parts = max(
            1,
            min(
                max(
                    min(
                        ceildiv(key_tiles, _KEY_TILES_PER_CTA),
                        _MAX_KEY_PART_WAVES * max(wave_parts, 1),
                    ),
                    wave_parts,
                ),
                key_tiles,
            ),
        )

    comptime kernel = _fp8_index_score_prefill_kernel_sm100[
        dtype,
        KOperand,
        KSOperand,
        type_of(valid_length.as_immut()).LayoutType,
        type_of(q_s).LayoutType,
        type_of(output).LayoutType,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
        kpool,
        VLEngine=type_of(valid_length.as_immut()).Engine,
        QSEngine=q_s.Engine,
        OutEngine=output.Engine,
    ]
    comptime CFG = IndexPrefillConfig[
        dtype,
        KSOperand.dtype,
        depth,
        BM_key,
        N_TOKENS * num_heads,
        num_heads,
    ]()
    # Validate where the config is BUILT, ahead of the derived asserts below --
    # the SMEM check fires on an over-forced ring depth first and reports only a
    # byte count, which hides which knob caused it.
    # The `setmaxnreg` caps redistribute a fixed file: 65536 registers/SM shared
    # by `CTAS_PER_SM` CTAs, and every thread of a warpgroup holds that
    # warpgroup's cap. Counted per thread rather than per warpgroup so it stays
    # right when the producer is not a full warpgroup.
    comptime assert (
        _NUM_SOFTMAX_THREADS * CFG.reg_consumer
        + 4 * WARP_SIZE * _NUM_REG_PRODUCER
        <= 65536 // CFG.ctas_per_sm
    ), (
        "the warpgroup register caps must fit the per-SM register file at"
        " CTAS_PER_SM CTAs/SM"
    )
    comptime smem_bytes = CFG.smem_used
    # Overrunning the per-SM budget does not fail the launch -- it just seats
    # fewer CTAs, so `CTAS_PER_SM` and the `minctasm` bound derived from it
    # would quietly become a lie. Fail at comptime instead. (~1KB/SM is reserved
    # by the driver, which is why the 1-CTA case caps at 227KB, not 228KB.)
    comptime assert (
        smem_bytes
        <= ctx.default_device_info.shared_memory_per_multiprocessor
        // CTAS_PER_SM
        - 1024
    ), (
        "prefill SMEM ("
        + String(smem_bytes)
        + " B) exceeds the budget for CTAS_PER_SM="
        + String(CTAS_PER_SM)
        + "; lower the K ring depth for MMA_N="
        + String(MMA_N)
    )
    # `max_num_keys` crosses the ABI as `Int32` and the kernel's whole key-index
    # chain is 32-bit signed, so the narrowing here is a silent truncation of a
    # caller-supplied `Int`. One key tile of headroom rather than stopping at
    # `Int32.MAX`: `key_local` already reaches one tile past `num_keys` on the
    # last tile of a non-BM-aligned range, and at `Int32.MAX` that wraps negative
    # and passes the signed `key_local < key_bound` store guard. It used to be
    # two tiles because the consumer speculatively gathered `k_scale` one tile
    # ahead; the k-scale TMA reads only the tile it is staging, so that extra
    # tile is no longer touched.
    comptime KEY_INDEX_HEADROOM = BM_key
    debug_assert[assert_mode="safe"](
        max_num_keys <= Int(Int32.MAX) - KEY_INDEX_HEADROOM,
        (
            "fp8 index prefill: max_num_keys must leave one key tile of"
            " headroom under Int32.MAX; the device-side key"
            " arithmetic is 32-bit signed"
        ),
    )
    # Built here, not beside `k_tma` in `fp8_index_score_sm100`: the K-resident
    # scorer needs no scale descriptor, and this is host work paid per launch.
    var ks_tma = ks_operand.create_index_scale_tma_tile[BM_key](ctx)
    ctx.enqueue_function[kernel](
        q_tma,
        k_tma,
        ks_tma,
        k_operand,
        ks_operand,
        valid_length.as_immut(),
        q_s,
        output,
        Int32(max_num_keys),
        Int32(causal),
        Int32(num_key_parts),
        Int32(out_row_begin),
        Int32(out_row_begin + out_rows),
        grid_dim=(
            batch_size,
            token_blocks,
            num_key_parts,
        ),
        block_dim=CFG.nthreads,
        shared_mem_bytes=smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        ),
    )
