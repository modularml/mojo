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
#
# Fuzz target: the single-launch MXFP8 MegaFFN (`mega_ffn_mxfp8_dispatch`) and
# the EP-combine peer send fused into its L2 epilogue.
#
# WHY THIS EXISTS. Every accuracy claim on the fused kernel today is
# `fused == unfused`, byte for byte. That is necessary and not sufficient: the
# two paths share the SwiGLU epilogue, the block-scaled quantizer and the
# scale-tile addressing, so a shared bug agrees with itself. This target adds a
# THIRD reference that shares no code with either path -- a host FP64 recompute
# of the whole chain (dequantize, GEMM, SwiGLU, requantize, GEMM, per-expert
# scale) -- and reports the fused-vs-unfused and both-vs-independent errors
# separately, because they answer different questions.
#
# ORACLE PER PATH (derived from the sources, not assumed):
#
#   1. FFN core, fused vs unfused -- BYTE-EXACT is the correct gate, and any
#      deviation is a bug. `mega_ffn_swiglu_mxfp8`'s `match_bf16=True` rounds
#      the SwiGLU epilogue through bf16 exactly where the two-launch chain
#      round-trips it through GMEM, and both entries pick (mma_bn, cta_group)
#      from the SAME `avg_m` cascade, so the per-tile K-block accumulation
#      order is identical. Both are driven here through their production
#      dispatch entries at one shared `estimated_total_m`, so the cascade
#      itself is under test rather than pinned around.
#
#   2. FFN core vs the independent reference -- TOLERANCE, not byte-exact. The
#      kernel accumulates in fp32 and the reference in fp64, and the chain
#      amplifies: an fp32-vs-fp64 accumulator difference of ~sqrt(K1)*2^-24
#      relative can straddle a bf16 rounding boundary in the L1 epilogue
#      (bf16 ULP 2^-8), which can in turn straddle an e4m3 boundary in the
#      intermediate (e4m3 ULP 2^-3), which perturbs one of K2 terms of the L2
#      dot product. Two gates, both derived before any measurement:
#        L2-only (reference fed the KERNEL's own intermediate): the only
#          remaining difference is fp64-vs-fp32 accumulation followed by ONE
#          bf16 rounding, so two roundings of the same real number differ by at
#          most one bf16 ULP -> rtol = 2^-8, no mismatch allowance.
#        end-to-end (fully independent chain): adds a possible 1-e4m3-ULP flip
#          of an intermediate element, diluted over the K2-long L2 dot product
#          -> rtol = 2^-5 with a bounded mismatch FRACTION, since the flip is a
#          per-element event whose rate is set by the boundary-crossing
#          probability, not by a magnitude bound.
#
#   3. Combine send, fused vs unfused vs independent -- BYTE-EXACT on all
#      three. The send moves bf16 bytes and does no arithmetic; the only thing
#      it computes is a destination address. The independent reference is a
#      host scatter built from the documented formula
#      (`(src_idx*top_k + src_topk)*msg_bytes`, `P3PeerSendConfig`), applied to
#      the kernel's own C. Unwritten receive slots must still hold the
#      sentinel, so the WRITTEN-SLOT SET is compared as well as the bytes: a
#      send that writes the right bytes to too many places is still wrong.
#
# EMPTY IS DECLARED, NEVER DISABLED. A combine send whose per-(expert, rank)
# token ranges are all zero moves nothing, so any comparison after it passes on
# untouched buffers. `_send_live_tokens` is the liveness witness (it sums the
# count halves of the dispatch-wait counter region the send resolves through)
# and `CaseSpec.empty_ok` declares, per case, whether zero is the intended
# condition. An undeclared empty is a vacuous gate and raises; a declared empty
# asserts the complementary invariant instead -- that the receive buffer is
# still ENTIRELY sentinel. Neither direction is silent.
#
# The send arm needs the in-epilogue send compiled in (`-D
# P5_DIRECT_SCATTER=true -D P5_ROW_CACHE=true`); it is default-off in the
# kernel, so a default build runs the send arm's unfused-vs-independent legs
# and reports the fused leg as not compiled rather than passing vacuously.
#
# Single GPU by construction: the `n_ranks` receive buffers are all local
# allocations, so every destination takes the same-node direct-store branch --
# the branch the fused epilogue send implements -- and a per-rank fan-out
# (including a rank that receives nothing) is expressible without extra GPUs.

from std.math import align_up, ceildiv, exp, max, min, sqrt
from std.utils.static_tuple import StaticTuple
from std.collections import Array
from std.memory import alloc
from std.random import rand, random_float64, random_ui64, seed
from std.sys import size_of
from std.sys.defines import get_defined_bool, get_defined_int

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_mxfp8_dispatch,
    grouped_matmul_swiglu_mxfp8_dispatch,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d.dispatch import (
    DECODE_AVG_M,
    SMALL_PREFILL_AVG_M,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.output_writer import (
    P3_MAX_RANKS,
)
from mega_ffn.mega_ffn_kernel import MODE_MEGAFFN
from mega_ffn.mega_ffn_matmul import (
    mega_ffn_mxfp8_dispatch,
    mega_ffn_swiglu_mxfp8,
)
from mega_ffn.mega_ffn_scheduler import (
    ARRIVAL_UNKNOWN,
    ATOMIC_PAD,
    POST_SELF_CLEAN_UP,
    assert_arrival_slots_after_launch,
)
from shmem.ep_comm import (
    EPLocalSyncCounters,
    combine_async_kernel,
)

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime a_type = DType.float8_e4m3fn
comptime c_type = DType.bfloat16
comptime sf_type = MXFP8_SF_DTYPE
comptime SF_VEC = MXFP8_SF_VECTOR_SIZE

# FFN geometry. Compile-time because the tuned SM100 kernel reads N/K from the
# tensors' STATIC shapes. The K constraint is the decode config's: both GEMM
# legs need `K/128 % 4 == 0`, and the L2's K is the intermediate width `N1/2`,
# so N1 must be a multiple of 1024 and K1 a multiple of 512.
comptime num_experts = get_defined_int["mff_num_experts", 8]()
comptime N1 = get_defined_int["mff_N1", 1024]()
comptime K1 = get_defined_int["mff_K1", 512]()
comptime N2 = get_defined_int["mff_N2", 512]()
comptime H = N1 // 2
comptime K2 = H

# EP geometry for the send arm. `n_ranks` receive buffers, all local.
comptime ep_n_ranks = get_defined_int["ep_n_ranks", 2]()
comptime ep_top_k = get_defined_int["ep_top_k", 4]()
comptime ep_max_tokens_per_rank = get_defined_int["ep_max_tokens", 256]()
comptime ep_n_experts = num_experts * ep_n_ranks
comptime msg_bytes = N2 * size_of[Scalar[c_type]]()
comptime recv_slots = ep_max_tokens_per_rank * ep_top_k
# Guard slots past the real receive capacity: a destination resolved one row
# past the end lands here instead of in another allocation, so the sentinel
# scan sees it even without a sanitizer.
comptime recv_guard_slots = 8
comptime recv_buf_slots = recv_slots + recv_guard_slots
comptime recv_capacity_rows = num_experts * ep_n_ranks * ep_max_tokens_per_rank

comptime k1_blocks = ceildiv(K1, SF_VEC)
comptime k2_blocks = ceildiv(K2, SF_VEC)
comptime k1_groups = ceildiv(K1, SF_VEC * SF_ATOM_K)
comptime k2_groups = ceildiv(K2, SF_VEC * SF_ATOM_K)
comptime n1_groups = ceildiv(N1, SF_MN_GROUP_SIZE)
comptime n2_groups = ceildiv(N2, SF_MN_GROUP_SIZE)
comptime w13_sf_atoms = (
    n1_groups * k1_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
)
comptime w2_sf_atoms = (
    n2_groups * k2_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
)

# Token-count ceilings. `MAX_TOTAL_TOKENS` keeps every case inside the receive
# capacity so the send arm's destinations stay resolvable; `REF_MAX_TOKENS`
# bounds the O(M*N1*K1) host recompute so the independent leg cannot time out
# and be misread as a kernel hang.
comptime MAX_TOTAL_TOKENS = get_defined_int["mff_max_tokens", 2048]()
comptime REF_MAX_TOKENS = get_defined_int["mff_ref_max_tokens", 1280]()

# The in-epilogue send is gated behind its own build knob in the kernel, so the
# fused leg is only real when that knob is on. Mirrored (not imported) for the
# same reason the kernel mirrors the src_info layout: this file must not depend
# on which of the two send mechanisms a build selected.
comptime send_compiled = get_defined_bool["P5_DIRECT_SCATTER", False]()
comptime send_row_cache = get_defined_bool["P5_ROW_CACHE", False]()

# Activation flavor. Both the fused kernel, the reference chain and the
# independent reference apply the same one, so the byte-exact leg is unaffected
# by the choice.
comptime clamp_act = get_defined_bool["mff_clamp", False]()
comptime SWIGLU_ALPHA = Float32(1.702)
comptime SWIGLU_LIMIT = Float32(7.0)

# Independent RNG streams. The probe construction for the batch-invariance
# oracle needs the probe's rows to be byte-identical across two compositions
# whose TOTAL token count differs, so each fill region starts from its own
# seed rather than from wherever the previous region left the stream.
comptime SEED_W = 0x2545_F491
comptime SEED_A = 0x1D2C_3E4F
comptime SEED_ASF = 0x7F4A_7C15

# Canary knob (`-D mff_poison=N`, 0 = off). Each value injects one defect a
# real bug could produce, so every comparison family can be shown to go red.
# 1, 3 and 4 poison the DEVICE side only (the host copy the independent
# reference reads is restored afterwards), which is what makes them a test of
# the reference rather than of the comparator.
#   1: one ACTIVE expert's W13 gate column is zeroed on the device
#                                                     -> the L1 intermediate
#                                                        leg red, and the C
#                                                        legs with it
#   2: one output element's exponent bit is flipped after readback
#                                                     -> byte-exact leg red and
#                                                        the send legs red
#   3: the L2 per-expert scale array is rotated by one on the device
#                                                     -> the L2-only leg red
#   4: one `src_info` row differs on the device       -> send legs red
#   5: the published token ranges are cleared         -> liveness witness fires
#   6: one ACTIVE expert's W2 output row is zeroed on the device
#                                                     -> the L2-only leg red
#   7: the batch-invariance probe's own token values differ between the two
#      compositions                                   -> the invariance oracle
#                                                        red (its positive
#                                                        control)
# 1, 3, 4 and 6 must target an ACTIVE expert: the routed `expert_ids` are a
# rotation, so a poison at expert slot 0 of the weight table is read by nobody
# and the canary passes while proving nothing.
# Reproduces the byte-uniform e4m3 fill the shipped MegaFFN MXFP8 test uses, so
# the cost of that choice is measurable rather than argued: 2 of 256 e4m3fn
# codes are non-finite, so a K1-long dot product is non-finite with probability
# `1 - (254/256)^K1` and a value comparison degenerates into comparing NaN
# patterns. Default off; `_tol_check`'s vacuity guard is what fires under it.
comptime byte_fill = get_defined_bool["mff_byte_fill", False]()

# Run the two-launch reference chain even at zero tokens. Default off (at zero
# tokens there is nothing to compare, and it would preempt the fused kernel's
# own zero-extent behaviour); on, it is the probe for the grouped family's
# early-out.
comptime ref_at_zero = get_defined_bool["mff_ref_at_zero", False]()

comptime POISON = get_defined_int["mff_poison", 0]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# E4M3 quantizer constants, from the fused L1 epilogue's own definitions. The
# reference implements the same SPECIFICATION (block amax -> E8M0 scale ->
# e4m3 elements); it shares no addressing or reduction code with it.
comptime E4M3_MAXABS_RECIP = Float32(1.0 / 448.0)

# ===----------------------------------------------------------------------=== #
# Shape families
# ===----------------------------------------------------------------------=== #
#
# The shipped tests exercise token counts 1, 2, 8 and a few round numbers, all
# comfortably inside a tile. These families exist to leave that neighbourhood
# deliberately: a partial last tile, a prime that divides no tile height, a
# count aligned to the value block but not the 128-row scale group, and the
# empty cases the ragged path can legitimately be handed at low batch.

comptime SC_RANDOM = 0
comptime SC_TILE_BOUNDARY = 1
comptime SC_PRIME = 2
comptime SC_SCALE_BOUNDARY = 3
comptime SC_ONE = 4
comptime SC_ZERO_ONE = 5
comptime SC_ZERO_ALL = 6
comptime SC_SKEW_SINGLE = 7
comptime SC_SKEW_RAGGED = 8
comptime SC_MASKED = 9
comptime NUM_SHAPE_CLASSES = 10

comptime ARM_FFN = 0
comptime ARM_SEND = 1


def shape_class_name(c: Int) -> String:
    if c == SC_TILE_BOUNDARY:
        return "tile_boundary"
    if c == SC_PRIME:
        return "prime"
    if c == SC_SCALE_BOUNDARY:
        return "scale_boundary"
    if c == SC_ONE:
        return "one"
    if c == SC_ZERO_ONE:
        return "zero_one"
    if c == SC_ZERO_ALL:
        return "zero_all"
    if c == SC_SKEW_SINGLE:
        return "skew_single"
    if c == SC_SKEW_RAGGED:
        return "skew_ragged"
    if c == SC_MASKED:
        return "masked"
    return "random"


def _primes() -> List[Int]:
    """Token counts that divide none of the tile heights (8, 64, 128) nor the
    32-element value block nor the 128-row scale group."""
    return [2, 3, 5, 7, 11, 13, 17, 31, 61, 127, 251, 509, 1021]


def _tile_heights() -> List[Int]:
    """Every `mma_bn` the `avg_m` cascade can select, decode first."""
    return [8, 64, 128]


def _regime_upper() -> List[Int]:
    """Per-tile-height upper bound on `avg_m`; -1 = the unbounded catch-all."""
    return [DECODE_AVG_M, SMALL_PREFILL_AVG_M, -1]


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var arm: Int
    var shape_class: Int
    var num_active_experts: Int
    var tok_seed: Int
    var empty_ok: Int
    var c_dead: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "arm=",
            self.arm,
            " shape_class=",
            self.shape_class,
            " num_active_experts=",
            self.num_active_experts,
            " tok_seed=",
            self.tok_seed,
            " empty_ok=",
            self.empty_ok,
            " c_dead=",
            self.c_dead,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var sc = boundary_int(0, NUM_SHAPE_CLASSES - 1, SC_TILE_BOUNDARY)
        var arm = ARM_SEND if random_ui64(0, 2) == 0 else ARM_FFN
        var nae = boundary_int(1, num_experts, num_experts)
        # `num_active_experts = 0` is only reachable deliberately: the whole
        # ragged index structure degenerates, so it belongs to the declared
        # empty family rather than the general draw.
        if sc == SC_ZERO_ALL and random_ui64(0, 1) == 0:
            nae = 0
        var empty_ok = 1 if (sc == SC_ZERO_ALL) else 0
        # The C-elision sub-arm only exists on the send arm (the epilogue
        # asserts that eliding the local store needs the peer send compiled in,
        # because the send then becomes C's only consumer).
        var c_dead = 1 if (arm == ARM_SEND and random_ui64(0, 2) == 0) else 0
        specs.append(
            CaseSpec(
                arm, sc, nae, Int(random_ui64(1, 1 << 30)), empty_ok, c_dead
            )
        )
    return specs^


# ===----------------------------------------------------------------------=== #
# Token distribution
# ===----------------------------------------------------------------------=== #


struct _Dist(Movable):
    """A case's expanded ragged token distribution.

    `counts[e]` tokens for expert slot `e`, split across EP ranks by
    `rank_counts[e * ep_n_ranks + rk]`. `ids[e]` is the routed expert id, or
    -1 for a masked slot (which always carries zero tokens).
    """

    var counts: List[Int]
    var ids: List[Int]
    var rank_counts: List[Int]
    var total: Int

    def __init__(
        out self,
        var counts: List[Int],
        var ids: List[Int],
        var rank_counts: List[Int],
    ):
        self.counts = counts^
        self.ids = ids^
        self.rank_counts = rank_counts^
        var t = 0
        for i in range(len(self.counts)):
            t += self.counts[i]
        self.total = t


def _split_across_ranks(
    counts: List[Int], shape_class: Int, tok_seed: Int
) -> List[Int]:
    """Split each expert's tokens across the EP ranks.

    The split is what the send resolves through, so the edge conditions live
    here: an all-local case (one rank owns everything) and a case where the
    LAST rank receives nothing at all.
    """
    var out = List[Int]()
    seed(tok_seed ^ 0x5BF0_3635)
    # Derived from the seed rather than drawn, so a spec names an exact split:
    # 0 = spread, 1 = all-local (rank 0 owns everything, later ranks receive
    # nothing), 2 = the LAST rank receives nothing.
    var mode = (tok_seed // 7) % 3
    if shape_class == SC_SKEW_SINGLE:
        mode = 1
    for e in range(len(counts)):
        var left = counts[e]
        for rk in range(ep_n_ranks):
            var take: Int
            if rk == ep_n_ranks - 1:
                take = left
            elif mode == 1:
                # All-local: rank 0 owns every token, later ranks receive none.
                take = left if rk == 0 else 0
            elif mode == 2:
                # Starve the last rank: spread over ranks [0, n_ranks-1).
                var denom = max(1, ep_n_ranks - 1 - rk)
                take = left // denom if rk < ep_n_ranks - 1 else 0
            else:
                take = Int(random_ui64(0, UInt64(max(0, left))))
            take = max(0, min(take, left))
            out.append(take)
            left -= take
        # `left` is 0 here: the last rank absorbs the remainder.
    return out^


def _expand(spec: CaseSpec) -> _Dist:
    """Deterministically expand `(shape_class, num_active_experts, tok_seed)`
    into per-expert and per-(expert, rank) token counts.

    Seeding happens at the top so `--mode single` reproduces the exact case
    the orchestrator recorded.
    """
    seed(spec.tok_seed)
    var nae = spec.num_active_experts
    var counts = List[Int]()
    var ids = List[Int]()
    var base = spec.tok_seed % max(1, num_experts)

    if spec.shape_class == SC_ZERO_ALL:
        for _ in range(nae):
            counts.append(0)
    elif spec.shape_class == SC_ONE:
        for i in range(nae):
            counts.append(1 if i == 0 else 0)
    elif spec.shape_class == SC_TILE_BOUNDARY:
        # `BM-1, BM, BM+1` for one selected tile height, with the remaining
        # slots sized so `avg_m = total/nae` lands in that height's regime.
        var heights = _tile_heights()
        var uppers = _regime_upper()
        # Derived from the seed, not drawn: seeds 0..8 enumerate all nine
        # (tile height, delta) pairs, so the triple is addressable.
        var r = (spec.tok_seed // 3) % 3
        var bm = heights[r]
        var delta = (spec.tok_seed % 3) - 1
        var head = max(0, bm + delta)
        counts.append(head)
        # Filler size: pick the per-expert count that pulls `avg_m` inside
        # regime `r` (below its upper bound, above the previous one).
        var lo = 1 if r == 0 else uppers[r - 1] + 1
        var hi = uppers[r] if uppers[r] > 0 else head
        var fill = max(1, min(hi, max(lo, head // 2)))
        for _ in range(1, nae):
            counts.append(fill)
    elif spec.shape_class == SC_PRIME:
        var ps = _primes()
        for i in range(nae):
            counts.append(ps[(spec.tok_seed + i) % len(ps)])
    elif spec.shape_class == SC_SCALE_BOUNDARY:
        # Counts that straddle the 32-element value block and the 128-row
        # scale group independently. 96 and 128 occupy the SAME number of
        # scale groups but leave different amounts of pad, which is the
        # asymmetry a value-block-only sweep never reaches.
        var cs: List[Int] = [
            31,
            32,
            33,
            95,
            96,
            97,
            127,
            128,
            129,
            159,
            160,
            161,
        ]
        for i in range(nae):
            counts.append(cs[(spec.tok_seed + i) % len(cs)])
    elif spec.shape_class == SC_ZERO_ONE:
        for i in range(nae):
            counts.append(0 if i == 0 else boundary_int(1, 160, 8))
    elif spec.shape_class == SC_SKEW_SINGLE:
        for i in range(nae):
            counts.append(boundary_int(1, 512, 128) if i == 0 else 0)
    elif spec.shape_class == SC_SKEW_RAGGED:
        for i in range(nae):
            counts.append(1 if (i % 2) else boundary_int(1, 256, 128))
    elif spec.shape_class == SC_MASKED:
        for i in range(nae):
            counts.append(0 if (i % 3 == 1) else boundary_int(1, 96, 8))
    else:
        for _ in range(nae):
            counts.append(boundary_int(1, 256, 8))

    for i in range(nae):
        # A masked slot carries no tokens: the fused send indexes the counter
        # region by expert id, so a masked (-1) slot with rows would resolve
        # through a negative index.
        var masked = spec.shape_class == SC_MASKED and (i % 3 == 1)
        if masked:
            counts[i] = 0
            ids.append(-1)
        else:
            ids.append((base + i) % num_experts)

    # Clamp the total, keeping the leading (boundary) expert intact so the
    # family's subject survives the clamp. The send arm needs a tighter cap: a
    # skewed split can route every token to ONE rank, and that rank's
    # destination slots are `max_tokens_per_rank * top_k`.
    var cap = (
        min(MAX_TOTAL_TOKENS, recv_slots) if spec.arm
        == ARM_SEND else MAX_TOTAL_TOKENS
    )
    var total = 0
    for i in range(len(counts)):
        total += counts[i]
    var i = len(counts) - 1
    while total > cap and i > 0:
        total -= counts[i]
        counts[i] = 0
        i -= 1
    if total > cap and len(counts) > 0:
        counts[0] = cap
    # Only the send arm is bounded per expert: the destinations it resolves
    # live in the EP receive buffer. Capping the FFN arm the same way would
    # silently truncate the largest boundary counts (1021 -> 512) and hide the
    # shapes the family exists to reach.
    if spec.arm == ARM_SEND:
        var per_expert_cap = ep_n_ranks * ep_max_tokens_per_rank
        for j in range(len(counts)):
            counts[j] = min(counts[j], per_expert_cap)

    var rank_counts = _split_across_ranks(
        counts, spec.shape_class, spec.tok_seed
    )
    return _Dist(counts^, ids^, rank_counts^)


def _selected_tile_height(total: Int, nae: Int) -> Int:
    """The `mma_bn` the production `avg_m` cascade selects for this shape.

    Reproduced (not imported) so the case log can name the tile height whose
    boundary a case actually landed on; the cascade itself stays under test
    because both the fused entry and the reference chain run it.
    """
    var heights = _tile_heights()
    var uppers = _regime_upper()
    for r in range(len(uppers)):
        if uppers[r] < 0 or total <= nae * uppers[r]:
            return heights[r]
    return heights[len(heights) - 1]


# ===----------------------------------------------------------------------=== #
# Comparators
# ===----------------------------------------------------------------------=== #


def _compare_bytes[
    name: StaticString
](
    lhs: UnsafePointer[UInt8, MutUntrackedOrigin],
    rhs: UnsafePointer[UInt8, MutUntrackedOrigin],
    size: Int,
) raises:
    """Byte-exact gate. Used where the arithmetic is identical by construction
    (fused vs unfused FFN) or absent (the send moves bytes)."""
    var mismatch = 0
    var first_bad = -1
    for i in range(size):
        if lhs[i] != rhs[i]:
            if first_bad < 0:
                first_bad = i
            mismatch += 1
    if mismatch != 0:
        print(
            "FUZZ_NUMERIC_FAIL kind=byte_exact name=",
            name,
            " n_bad=",
            mismatch,
            " first_bad=",
            first_bad,
            " n=",
            size,
            sep="",
        )
        raise Error(String("byte-exact mismatch: ") + String(name))


def _tol_check[
    name: StaticString
](
    actual: UnsafePointer[Scalar[c_type], MutUntrackedOrigin],
    expected: UnsafePointer[Float64, MutUntrackedOrigin],
    n: Int,
    atol: Float64,
    rtol: Float64,
    allow_frac: Float64,
) raises:
    """Tolerance gate against the independent FP64 reference.

    Always prints the observed error, pass or fail, so the three-way numbers
    are reported rather than hidden behind a verdict. `allow_frac` bounds the
    FRACTION of elements allowed to miss: an intermediate e4m3 rounding flip is
    a per-element event with a boundary-crossing rate, so a magnitude-only
    bound would have to be uselessly loose to cover it.
    """
    if n == 0:
        print("FUZZ_ERR name=", name, " n=0 (skipped)", sep="")
        return
    var n_bad = 0
    var max_abs = Float64(0)
    var max_rel = Float64(0)
    var worst = -1
    var n_exp_nonfinite = 0
    var n_act_nonfinite = 0
    var n_useful = 0
    for i in range(n):
        var e = expected[i]
        var a = actual[i].cast[DType.float64]()
        var e_ok = (e == e) and (e - e == Float64(0))
        var a_ok = (a == a) and (a - a == Float64(0))
        if not e_ok:
            n_exp_nonfinite += 1
            continue
        if not a_ok:
            n_act_nonfinite += 1
            n_bad += 1
            if worst < 0:
                worst = i
            continue
        if e != Float64(0):
            n_useful += 1
        # Track the max over EVERY element, not just the failing ones: the
        # observed error is the number worth reporting even on a pass.
        var ad = abs(a - e)
        var rel = ad / (abs(e) + 1e-300)
        if ad > max_abs:
            max_abs = ad
            worst = i
        if rel > max_rel:
            max_rel = rel
        if ad > atol + rtol * abs(e):
            n_bad += 1
    # Vacuity guard: a comparison whose reference is mostly non-finite or
    # mostly zero cannot distinguish a correct kernel from a dead one.
    if n_exp_nonfinite * 2 > n:
        raise Error(
            String("VACUOUS COMPARE (")
            + String(name)
            + "): "
            + String(n_exp_nonfinite)
            + " of "
            + String(n)
            + " reference elements are non-finite, so the tolerance gate is"
            " comparing NaN to NaN. The input fill must stay inside the finite"
            " e4m3 range."
        )
    if n_useful * 4 < n:
        raise Error(
            String("VACUOUS COMPARE (")
            + String(name)
            + "): only "
            + String(n_useful)
            + " of "
            + String(n)
            + " reference elements are nonzero."
        )
    print(
        "FUZZ_ERR name=",
        name,
        " n=",
        n,
        " n_bad=",
        n_bad,
        " frac_bad=",
        Float64(n_bad) / Float64(n),
        " max_abs=",
        max_abs,
        " max_rel=",
        max_rel,
        " worst=",
        worst,
        " exp_nonfinite=",
        n_exp_nonfinite,
        " act_nonfinite=",
        n_act_nonfinite,
        " atol=",
        atol,
        " rtol=",
        rtol,
        sep="",
    )
    if Float64(n_bad) > allow_frac * Float64(n):
        print(
            "FUZZ_NUMERIC_FAIL kind=tolerance name=",
            name,
            " n_bad=",
            n_bad,
            " allowed=",
            allow_frac * Float64(n),
            " max_rel=",
            max_rel,
            sep="",
        )
        raise Error(String("independent-reference mismatch: ") + String(name))


# ===----------------------------------------------------------------------=== #
# Independent reference (shares no code with either kernel path)
# ===----------------------------------------------------------------------=== #
#
# Reads plain row-major mirrors of the block scales that the setup fills
# alongside the swizzled scale-factor tiles, so nothing here depends on the
# scale-tile addressing that both kernel paths share.


def _ref_l1(
    a: UnsafePointer[Scalar[a_type], MutUntrackedOrigin],
    a_sf: UnsafePointer[Float32, MutUntrackedOrigin],
    w13: UnsafePointer[Scalar[a_type], MutUntrackedOrigin],
    w13_sf: UnsafePointer[Float32, MutUntrackedOrigin],
    es1: UnsafePointer[Float32, MutUntrackedOrigin],
    counts: List[Int],
    ids: List[Int],
    offs: List[Int],
    sf_row: List[Int],
    clamp: Bool,
    alpha: Float32,
    limit: Float32,
    u_out: UnsafePointer[Float64, MutUntrackedOrigin],
):
    """Block-scaled A@W13 in FP64, then the fused epilogue's activation.

    `w13` is already in the gate/up-interleaved order the fused epilogue
    expects, so column `2i` is the gate and `2i+1` the up for output `i`. The
    per-expert L1 scale multiplies the accumulator BEFORE the bf16 round and
    therefore before the activation, which is not the same as scaling the
    activation's result.
    """
    var pre = alloc[Float64](N1)
    for e in range(len(counts)):
        var eid = ids[e]
        if counts[e] == 0 or eid < 0:
            continue
        var s1 = es1[eid]
        for local in range(counts[e]):
            var m = offs[e] + local
            var sf_r = sf_row[e] + local
            for n in range(N1):
                var acc = Float64(0)
                for kb in range(k1_blocks):
                    var k_lo = kb * SF_VEC
                    var k_hi = min(K1, k_lo + SF_VEC)
                    var blk = Float64(0)
                    for k in range(k_lo, k_hi):
                        blk += (
                            a[m * K1 + k].cast[DType.float64]()
                            * w13[(eid * N1 + n) * K1 + k].cast[DType.float64]()
                        )
                    acc += (
                        blk
                        * a_sf[sf_r * k1_blocks + kb].cast[DType.float64]()
                        * w13_sf[(eid * N1 + n) * k1_blocks + kb].cast[
                            DType.float64
                        ]()
                    )
                # The epilogue scales in fp32 and rounds through bf16, and
                # `match_bf16` makes that rounding part of the contract.
                pre[n] = (
                    (acc.cast[DType.float32]() * s1)
                    .cast[c_type]()
                    .cast[DType.float64]()
                )
            for h in range(H):
                var g = pre[2 * h]
                var u = pre[2 * h + 1]
                if clamp:
                    var lim = limit.cast[DType.float64]()
                    var g_c = min(g, lim)
                    var u_c = max(min(u, lim), -lim)
                    var sig = 1.0 / (
                        1.0 + exp(-(g_c * alpha.cast[DType.float64]()))
                    )
                    u_out[m * H + h] = (u_c + 1.0) * g_c * sig
                else:
                    var sig = 1.0 / (1.0 + exp(-g))
                    u_out[m * H + h] = g * sig * u
    pre.free()


def _ref_quant(
    u: UnsafePointer[Float64, MutUntrackedOrigin],
    total: Int,
    oq: UnsafePointer[Scalar[a_type], MutUntrackedOrigin],
    sq: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """MXFP8 requantize of the intermediate: per 32-element block along the
    intermediate width, one E8M0 scale from the block amax, then e4m3
    elements. The block amax and the `1/448` scale derivation are the fused
    epilogue's documented quantizer, reimplemented here.
    """
    for m in range(total):
        for hb in range(k2_blocks):
            var h_lo = hb * SF_VEC
            var h_hi = min(H, h_lo + SF_VEC)
            var bmax = Float32(0)
            for h in range(h_lo, h_hi):
                var v = abs(u[m * H + h]).cast[DType.float32]()
                if v > bmax:
                    bmax = v
            var sf_byte = (bmax * E4M3_MAXABS_RECIP).cast[sf_type]()
            var sf_val = sf_byte.cast[DType.float32]()
            var out_scale = Float32(0)
            if bmax != Float32(0):
                out_scale = Float32(1) / sf_val
            sq[m * k2_blocks + hb] = sf_val
            for h in range(h_lo, h_hi):
                oq[m * H + h] = (
                    u[m * H + h].cast[DType.float32]() * out_scale
                ).cast[a_type]()


def _ref_l2(
    oq: UnsafePointer[Scalar[a_type], MutUntrackedOrigin],
    sq: UnsafePointer[Float32, MutUntrackedOrigin],
    w2: UnsafePointer[Scalar[a_type], MutUntrackedOrigin],
    w2_sf: UnsafePointer[Float32, MutUntrackedOrigin],
    es2: UnsafePointer[Float32, MutUntrackedOrigin],
    counts: List[Int],
    ids: List[Int],
    offs: List[Int],
    c_out: UnsafePointer[Float64, MutUntrackedOrigin],
):
    """Block-scaled intermediate@W2 in FP64, scaled by the per-expert L2
    scale. Left un-rounded: the caller compares the kernel's bf16 output
    against this exact value, so the output rounding is inside the tolerance
    budget rather than replicated."""
    for e in range(len(counts)):
        var eid = ids[e]
        if counts[e] == 0 or eid < 0:
            continue
        var s2 = es2[eid].cast[DType.float64]()
        for local in range(counts[e]):
            var m = offs[e] + local
            for n in range(N2):
                var acc = Float64(0)
                for kb in range(k2_blocks):
                    var h_lo = kb * SF_VEC
                    var h_hi = min(K2, h_lo + SF_VEC)
                    var blk = Float64(0)
                    for h in range(h_lo, h_hi):
                        blk += (
                            oq[m * H + h].cast[DType.float64]()
                            * w2[(eid * N2 + n) * K2 + h].cast[DType.float64]()
                        )
                    acc += (
                        blk
                        * sq[m * k2_blocks + kb].cast[DType.float64]()
                        * w2_sf[(eid * N2 + n) * k2_blocks + kb].cast[
                            DType.float64
                        ]()
                    )
                c_out[m * N2 + n] = acc * s2


# ===----------------------------------------------------------------------=== #
# Case setup + the three-way comparison
# ===----------------------------------------------------------------------=== #


def _e8m0(x: Float32) -> Scalar[sf_type]:
    return x.cast[sf_type]()


def _rand_pow2_sf() -> Float32:
    """A block scale drawn from {1, 2, 4}.

    A single repeated scale makes the whole scale path degenerate: a kernel
    that dropped the block scales entirely would still match. Powers of two
    keep the scale multiply exact so it contributes no reference error.
    """
    return Float32(1 << Int(random_ui64(0, 2)))


def _send_live_tokens(
    ctx: DeviceContext, counters_dev: DeviceBuffer[DType.int32]
) raises -> Int:
    """Liveness witness for the combine send: the total token count published
    across the per-(expert, rank) ranges the send resolves through.

    Zero means the send moves nothing, so every comparison after it would pass
    on untouched buffers. `CaseSpec.empty_ok` decides whether that is the
    declared condition or a vacuous gate.

    Read from the DEVICE, not from the host mirror the harness wrote. That is
    not a detail: the standalone combine CONSUMES these ranges (it zeroes each
    per-(expert, rank) entry as it drains it), so a witness that trusts the
    host copy reports the ranges as live after they are gone, and the fused
    send that follows resolves no destination for any row and writes nothing.
    A host-side witness cannot see that; this one can.
    """
    comptime region_a = EPLocalSyncCounters[ep_n_experts].dispatch_async_size()
    comptime n = EPLocalSyncCounters[ep_n_experts].total_size()
    var host = alloc[Int32](n)
    ctx.enqueue_copy(host, counters_dev)
    ctx.synchronize()
    var live = 0
    for i in range(ep_n_experts):
        live += Int(host[region_a + 2 * i + 1])
    host.free()
    return live


def run_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    rerun: Int = 0,
    # Batch-invariance probe. When `probe_m > 0`, expert slot 0 carries exactly
    # `probe_m` tokens whose values come from the case seed while every filler
    # row comes from `filler_seed`, and the probe's output rows are copied out.
    probe_m: Int = 0,
    n_fillers: Int = 0,
    filler_m: Int = 0,
    filler_seed: Int = 0,
    probe_out: Optional[UnsafePointer[Scalar[c_type], MutUntrackedOrigin]] = (
        None
    ),
    # Positive control for the invariance oracle: perturbs the PROBE's own
    # token values in one composition, leaving the weights and the ragged
    # structure alone, so a divergence can only come from the probe's input.
    probe_seed_bump: Int = 0,
) raises:
    var dist = _expand(spec)
    var counts = dist.counts.copy()
    var rank_counts = dist.rank_counts.copy()
    if probe_m > 0:
        counts = [probe_m]
        for _ in range(n_fillers):
            counts.append(filler_m)
        rank_counts = _split_across_ranks(counts, spec.shape_class, filler_seed)
    var nae = len(counts)
    var total = 0
    for i in range(nae):
        total += counts[i]

    # The fused send indexes the per-(expert, rank) counter region by EXPERT
    # ID, while the standalone combine indexes it by local expert SLOT. Those
    # agree in production because `dispatch_wait` writes the identity mapping,
    # so the send arm uses identity ids; the FFN arm keeps a rotated mapping,
    # which exercises non-contiguous weight and scale slices.
    var ids = List[Int]()
    var id_base = spec.tok_seed % max(1, num_experts)
    for i in range(nae):
        if probe_m > 0:
            # Slot 0 must route to the SAME expert in both compositions, so
            # the id is a function of the case seed only.
            ids.append((id_base + i) % num_experts)
        elif dist.ids[i] < 0:
            ids.append(-1)
        elif spec.arm == ARM_SEND:
            ids.append(i)
        else:
            ids.append(dist.ids[i])

    var mma_bn = 8 if spec.arm == ARM_SEND else _selected_tile_height(
        total, max(1, nae)
    )
    # The send is only implemented for the decode tile geometry, so the send
    # arm pins both the fused launch and the reference chain there by feeding
    # the chain a decode-regime estimate. `estimated_total_m` selects geometry
    # and nothing else, so pinning it changes no arithmetic.
    var est = nae * DECODE_AVG_M if spec.arm == ARM_SEND else total

    var m_blocks = 0
    for i in range(nae):
        if ids[i] >= 0 and counts[i] > 0:
            m_blocks += ceildiv(counts[i], mma_bn)

    print(
        "  case ",
        shape_class_name(spec.shape_class),
        " arm=",
        "send" if spec.arm == ARM_SEND else "ffn",
        " nae=",
        nae,
        " total=",
        total,
        " mma_bn=",
        mma_bn,
        " m_blocks=",
        m_blocks,
        " c_dead=",
        spec.c_dead,
        sep="",
    )

    # ---- ragged index structure ----
    var a_offsets_host = alloc[Scalar[DType.uint32]](num_experts + 1)
    var a_scale_offsets_host = alloc[Scalar[DType.uint32]](num_experts)
    var expert_ids_host = alloc[Scalar[DType.int32]](num_experts)
    var offs = List[Int]()
    var sf_row = List[Int]()

    var a_scale_dim0 = 0
    a_offsets_host[0] = 0
    for i in range(nae):
        a_scale_offsets_host[i] = UInt32(
            a_scale_dim0 - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        offs.append(Int(a_offsets_host[i]))
        sf_row.append(
            (
                Int(a_offsets_host[i]) // SF_MN_GROUP_SIZE
                + Int(a_scale_offsets_host[i])
            )
            * SF_MN_GROUP_SIZE
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(counts[i])
        a_scale_dim0 += ceildiv(counts[i], SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(ids[i])
    for i in range(nae, num_experts):
        a_offsets_host[i + 1] = a_offsets_host[max(0, nae)]
        a_scale_offsets_host[i] = UInt32(0)
        expert_ids_host[i] = Int32(-1)
    if nae == 0:
        a_offsets_host[0] = 0

    # ---- weights (already gate/up-interleaved; no permutation step) ----
    comptime w13_size = num_experts * N1 * K1
    comptime w2_size = num_experts * N2 * K2
    seed(spec.tok_seed ^ SEED_W)
    var w13_host = alloc[Scalar[a_type]](w13_size)
    var w2_host = alloc[Scalar[a_type]](w2_size)
    # Values in [-1, 1] rather than random bytes: a uniform BYTE fill of e4m3
    # puts a NaN (0x7F / 0xFF) at 2/256 of positions, which over a K-long dot
    # product makes essentially every output NaN and turns any value
    # comparison into NaN-vs-NaN.
    comptime if byte_fill:
        rand(w13_host.bitcast[UInt8](), w13_size, min=0, max=255)
        rand(w2_host.bitcast[UInt8](), w2_size, min=0, max=255)
    else:
        for i in range(w13_size):
            w13_host[i] = random_float64(-1.0, 1.0).cast[a_type]()
        for i in range(w2_size):
            w2_host[i] = random_float64(-1.0, 1.0).cast[a_type]()

    var w13_sf_plain = alloc[Float32](num_experts * N1 * k1_blocks)
    var w2_sf_plain = alloc[Float32](num_experts * N2 * k2_blocks)
    var w13_scales_host = alloc[Scalar[sf_type]](num_experts * w13_sf_atoms)
    var w2_scales_host = alloc[Scalar[sf_type]](num_experts * w2_sf_atoms)
    var sf_zero = _e8m0(Float32(0.0))

    for e in range(num_experts):
        var view = TileTensor(
            w13_scales_host + e * w13_sf_atoms,
            row_major(
                Coord(
                    Idx[n1_groups],
                    Idx[k1_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(N1, SF_MN_GROUP_SIZE)):
            for idx1 in range(0, align_up(K1, SF_VEC * SF_ATOM_K), SF_VEC):
                var v = sf_zero
                if idx0 < N1 and idx1 < K1:
                    var f = _rand_pow2_sf()
                    v = _e8m0(f)
                    w13_sf_plain[
                        (e * N1 + idx0) * k1_blocks + idx1 // SF_VEC
                    ] = f
                set_scale_factor[SF_VECTOR_SIZE=SF_VEC](view, idx0, idx1, v)

    for e in range(num_experts):
        var view = TileTensor(
            w2_scales_host + e * w2_sf_atoms,
            row_major(
                Coord(
                    Idx[n2_groups],
                    Idx[k2_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(N2, SF_MN_GROUP_SIZE)):
            for idx1 in range(0, align_up(K2, SF_VEC * SF_ATOM_K), SF_VEC):
                var v = sf_zero
                if idx0 < N2 and idx1 < K2:
                    var f = _rand_pow2_sf()
                    v = _e8m0(f)
                    w2_sf_plain[
                        (e * N2 + idx0) * k2_blocks + idx1 // SF_VEC
                    ] = f
                set_scale_factor[SF_VECTOR_SIZE=SF_VEC](view, idx0, idx1, v)

    var es1_host = alloc[Float32](num_experts)
    var es2_host = alloc[Float32](num_experts)
    for i in range(num_experts):
        es1_host[i] = 1.0 + Float32(i + 1) / Float32(num_experts)
        # Distinct from the L1 map: a kernel that applied one per-expert scale
        # to both legs would still match a reference built from one array.
        es2_host[i] = 0.5 + 2.0 * Float32(i + 1) / Float32(num_experts)

    # ---- tokens ----
    var a_size = total * K1
    var a_host = alloc[Scalar[a_type]](max(1, a_size))
    seed(spec.tok_seed ^ SEED_A ^ probe_seed_bump)
    var probe_rows_end = probe_m * K1
    comptime if byte_fill:
        rand(a_host.bitcast[UInt8](), a_size, min=0, max=255)
    else:
        for i in range(a_size):
            if probe_m > 0 and i == probe_rows_end:
                seed(filler_seed ^ SEED_A)
            a_host[i] = random_float64(-1.0, 1.0).cast[a_type]()

    var a_sf_rows = max(1, a_scale_dim0 * SF_MN_GROUP_SIZE)
    var a_sf_plain = alloc[Float32](a_sf_rows * k1_blocks)
    for i in range(a_sf_rows * k1_blocks):
        a_sf_plain[i] = Float32(0)

    var a_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k1_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var a_scales_total = a_scales_shape.product()
    var a_scales_host = alloc[Scalar[sf_type]](max(1, a_scales_total))
    for i in range(a_scales_total):
        a_scales_host[i] = sf_zero
    var a_scales_view = TileTensor(a_scales_host, a_scales_shape)
    seed(spec.tok_seed ^ SEED_ASF)
    for i in range(nae):
        if probe_m > 0 and i == 1:
            seed(filler_seed ^ SEED_ASF)
        for local in range(counts[i]):
            var r = sf_row[i] + local
            for idx1 in range(0, K1, SF_VEC):
                var f = _rand_pow2_sf()
                a_sf_plain[r * k1_blocks + idx1 // SF_VEC] = f
                set_scale_factor[SF_VECTOR_SIZE=SF_VEC](
                    a_scales_view, r, idx1, _e8m0(f)
                )

    # ---- device side ----
    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var a_scales_dev = ctx.enqueue_create_buffer[sf_type](a_scales_total)
    var w13_dev = ctx.enqueue_create_buffer[a_type](w13_size)
    var w13_scales_dev = ctx.enqueue_create_buffer[sf_type](
        num_experts * w13_sf_atoms
    )
    var w2_dev = ctx.enqueue_create_buffer[a_type](w2_size)
    var w2_scales_dev = ctx.enqueue_create_buffer[sf_type](
        num_experts * w2_sf_atoms
    )
    var es1_dev = ctx.enqueue_create_buffer[DType.float32](num_experts)
    var es2_dev = ctx.enqueue_create_buffer[DType.float32](num_experts)
    var a_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](num_experts + 1)
    var a_scale_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        num_experts
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[DType.int32](num_experts)

    var s_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k2_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var s_total = s_shape.product()
    var o_size = total * K2
    var c_size = total * N2

    var o_ref_dev = ctx.enqueue_create_buffer[a_type](o_size)
    var o_test_dev = ctx.enqueue_create_buffer[a_type](o_size)
    var s_ref_dev = ctx.enqueue_create_buffer[sf_type](s_total)
    var s_test_dev = ctx.enqueue_create_buffer[sf_type](s_total)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_test_dev = ctx.enqueue_create_buffer[c_type](c_size)

    var ac_len = (m_blocks + 1) * ATOMIC_PAD
    var arrival_dev = ctx.enqueue_create_buffer[DType.uint32](ac_len)
    ctx.enqueue_memset(arrival_dev, UInt32(0))

    # Distinct garbage per path so an unwritten cell shows up as a mismatch
    # rather than as agreement.
    ctx.enqueue_memset(o_ref_dev, Scalar[a_type](from_bits=UInt8(0xCC)))
    ctx.enqueue_memset(o_test_dev, Scalar[a_type](from_bits=UInt8(0x33)))
    ctx.enqueue_memset(s_ref_dev, Scalar[sf_type](from_bits=UInt8(0xAA)))
    ctx.enqueue_memset(s_test_dev, Scalar[sf_type](from_bits=UInt8(0x55)))
    ctx.enqueue_memset(c_ref_dev, Scalar[c_type](1234.5))
    ctx.enqueue_memset(c_test_dev, Scalar[c_type](-777.25))

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    var poison_expert = -1
    for i in range(nae):
        if ids[i] >= 0 and counts[i] > 0:
            poison_expert = ids[i]
            break
    comptime if POISON == 1:
        if poison_expert >= 0:
            var saved = alloc[Scalar[a_type]](K1)
            for k in range(K1):
                saved[k] = w13_host[(poison_expert * N1) * K1 + k]
                w13_host[(poison_expert * N1) * K1 + k] = Scalar[a_type](0)
            ctx.enqueue_copy(w13_dev, w13_host)
            ctx.synchronize()
            for k in range(K1):
                w13_host[(poison_expert * N1) * K1 + k] = saved[k]
            saved.free()
        else:
            ctx.enqueue_copy(w13_dev, w13_host)
    else:
        ctx.enqueue_copy(w13_dev, w13_host)
    ctx.enqueue_copy(w13_scales_dev, w13_scales_host)
    comptime if POISON == 6:
        if poison_expert >= 0:
            var saved2 = alloc[Scalar[a_type]](K2)
            for h in range(K2):
                saved2[h] = w2_host[(poison_expert * N2) * K2 + h]
                w2_host[(poison_expert * N2) * K2 + h] = Scalar[a_type](0)
            ctx.enqueue_copy(w2_dev, w2_host)
            ctx.synchronize()
            for h in range(K2):
                w2_host[(poison_expert * N2) * K2 + h] = saved2[h]
            saved2.free()
        else:
            ctx.enqueue_copy(w2_dev, w2_host)
    else:
        ctx.enqueue_copy(w2_dev, w2_host)
    ctx.enqueue_copy(w2_scales_dev, w2_scales_host)
    ctx.enqueue_copy(es1_dev, es1_host)
    comptime if POISON == 3:
        var rot = alloc[Float32](num_experts)
        for i in range(num_experts):
            rot[i] = es2_host[(i + 1) % num_experts]
        ctx.enqueue_copy(es2_dev, rot)
        ctx.synchronize()
        rot.free()
    else:
        ctx.enqueue_copy(es2_dev, es2_host)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(a_scale_offsets_dev, a_scale_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)

    var a_tt = TileTensor(a_dev, row_major(Coord(Int(total), Idx[K1])))
    var a_scales_tt = TileTensor(
        a_scales_dev, a_scales_shape
    ).as_unsafe_any_origin()
    var w13_tt = TileTensor(
        w13_dev, row_major(Coord(Idx[num_experts], Idx[N1], Idx[K1]))
    )
    var w13_scales_tt = TileTensor(
        w13_scales_dev,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n1_groups],
                Idx[k1_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var w2_tt = TileTensor(
        w2_dev, row_major(Coord(Idx[num_experts], Idx[N2], Idx[K2]))
    )
    var w2_scales_tt = TileTensor(
        w2_scales_dev,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n2_groups],
                Idx[k2_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var es1_tt = TileTensor(
        es1_dev, row_major(Coord(Idx[num_experts]))
    ).as_unsafe_any_origin()
    var es2_tt = TileTensor(
        es2_dev, row_major(Coord(Idx[num_experts]))
    ).as_unsafe_any_origin()
    var a_offsets_tt = TileTensor(
        a_offsets_dev, row_major(Coord(Idx[num_experts + 1]))
    )
    var a_scale_offsets_tt = TileTensor(
        a_scale_offsets_dev, row_major(Coord(Idx[num_experts]))
    )
    var expert_ids_tt = TileTensor(
        expert_ids_dev, row_major(Coord(Idx[num_experts]))
    )
    var o_ref_tt = TileTensor(o_ref_dev, row_major(Coord(Int(total), Idx[K2])))
    var o_test_tt = TileTensor(
        o_test_dev, row_major(Coord(Int(total), Idx[K2]))
    )
    var s_ref_tt = TileTensor(s_ref_dev, s_shape)
    var s_test_tt = TileTensor(s_test_dev, s_shape)
    var c_ref_tt = TileTensor(c_ref_dev, row_major(Coord(Int(total), Idx[N2])))
    var c_test_tt = TileTensor(
        c_test_dev, row_major(Coord(Int(total), Idx[N2]))
    )

    if total == 0:
        print(
            "  FUZZ_NOTE zero_token_case empty_ok=",
            spec.empty_ok,
            (
                " -- the tensors below have a zero global extent, so any launch"
                " failure reported after this line is the kernel's zero-extent"
                " behaviour, not a harness fault"
            ),
            sep="",
        )

    # ---- unfused reference: the two-launch grouped chain ----
    # Skipped at zero tokens: there is nothing to compare, and running it
    # would reach the grouped launcher before the fused kernel is ever
    # launched, which hides what the fused kernel itself does with a zero
    # global extent. `-D mff_ref_at_zero=true` un-skips it, which is how the
    # grouped family's own zero-token early-out gets exercised (with the skip
    # in place, nothing in this target ever calls the grouped launcher at zero
    # tokens, so its guard would be present but unproven).
    if total > 0 or ref_at_zero:
        grouped_matmul_swiglu_mxfp8_dispatch[
            transpose_b=True,
            match_bf16=True,
            use_inplace=True,
            clamp_activation=clamp_act,
        ](
            o_ref_tt,
            s_ref_tt,
            a_tt,
            w13_tt,
            a_scales_tt,
            w13_scales_tt,
            a_offsets_tt,
            a_scale_offsets_tt,
            expert_ids_tt,
            es1_tt,
            nae,
            est,
            ctx,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
        )
        var o_ref_any = o_ref_tt.as_unsafe_any_origin()
        var s_ref_any = s_ref_tt.as_unsafe_any_origin()
        grouped_matmul_mxfp8_dispatch[transpose_b=True](
            c_ref_tt,
            o_ref_any,
            w2_tt,
            s_ref_any,
            w2_scales_tt,
            a_offsets_tt,
            a_scale_offsets_tt,
            expert_ids_tt,
            es2_tt,
            nae,
            est,
            ctx,
        )

    # ---- EP state for the send arm ----
    comptime counters_len = EPLocalSyncCounters[ep_n_experts].total_size()
    comptime region_a = EPLocalSyncCounters[ep_n_experts].dispatch_async_size()
    # `dispatch_wait` publishes each range's END with this flag added, and the
    # send subtracts it back out; the same constant appears in the combine
    # kernel as `DATA_READY_FLAG`.
    comptime DATA_READY_FLAG = 1024
    comptime SENTINEL = UInt8(0xAB)

    var counters_host = alloc[Int32](counters_len)
    var src_info_host = alloc[Int32](recv_capacity_rows * 2)
    var counters_dev = ctx.enqueue_create_buffer[DType.int32](counters_len)
    var src_info_dev = ctx.enqueue_create_buffer[DType.int32](
        recv_capacity_rows * 2
    )
    var row_base_dev = ctx.enqueue_create_buffer[DType.uint64](
        recv_capacity_rows
    )
    var send_stage_dev = ctx.enqueue_create_buffer[DType.uint8](
        recv_capacity_rows * msg_bytes
    )
    var recv_fused = List[DeviceBuffer[DType.uint8]](capacity=ep_n_ranks)
    var recv_unfused = List[DeviceBuffer[DType.uint8]](capacity=ep_n_ranks)
    var recv_count = List[DeviceBuffer[DType.uint64]](capacity=ep_n_ranks)
    for _ in range(ep_n_ranks):
        recv_fused.append(
            ctx.enqueue_create_buffer[DType.uint8](recv_buf_slots * msg_bytes)
        )
        recv_unfused.append(
            ctx.enqueue_create_buffer[DType.uint8](recv_buf_slots * msg_bytes)
        )
        recv_count.append(
            ctx.enqueue_create_buffer[DType.uint64](num_experts * ep_n_ranks)
        )

    if spec.arm == ARM_SEND:
        for i in range(counters_len):
            counters_host[i] = Int32(0)
        for i in range(recv_capacity_rows * 2):
            src_info_host[i] = Int32(-1)
        var next_slot = List[Int]()
        for _ in range(ep_n_ranks):
            next_slot.append(0)
        for e in range(nae):
            if ids[e] < 0:
                continue
            var cur = offs[e]
            for rk in range(ep_n_ranks):
                var cnt = rank_counts[e * ep_n_ranks + rk]
                var t_end = cur + cnt
                var pair = ids[e] * ep_n_ranks + rk
                counters_host[region_a + 2 * pair] = Int32(
                    t_end + DATA_READY_FLAG
                )
                counters_host[region_a + 2 * pair + 1] = Int32(cnt)
                for t in range(cnt):
                    var slot = next_slot[rk]
                    next_slot[rk] = slot + 1
                    src_info_host[(cur + t) * 2] = Int32(slot // ep_top_k)
                    src_info_host[(cur + t) * 2 + 1] = Int32(slot % ep_top_k)
                cur = t_end
        ctx.enqueue_copy(counters_dev, counters_host)
        comptime if POISON == 4:
            var saved_si = src_info_host[0]
            src_info_host[0] = (saved_si + Int32(1)) % Int32(
                ep_max_tokens_per_rank
            )
            ctx.enqueue_copy(src_info_dev, src_info_host)
            ctx.synchronize()
            src_info_host[0] = saved_si
        else:
            ctx.enqueue_copy(src_info_dev, src_info_host)
        ctx.enqueue_memset(row_base_dev, UInt64(0))
        for rk in range(ep_n_ranks):
            ctx.enqueue_memset(recv_fused[rk], SENTINEL)
            ctx.enqueue_memset(recv_unfused[rk], SENTINEL)
            ctx.enqueue_memset(recv_count[rk], UInt64.MAX_FINITE)
        ctx.synchronize()

        comptime if POISON == 5:
            ctx.enqueue_memset(counters_dev, Int32(0))
            ctx.synchronize()
        var live = _send_live_tokens(ctx, counters_dev)
        print("    send live tokens=", live, sep="")
        if live <= 0 and spec.empty_ok == 0:
            raise Error(
                "VACUOUS GATE: the combine send has no token ranges, so it"
                " moves nothing and every comparison after it would pass on"
                " untouched buffers. This case did not declare empty_ok=1, so"
                " the empty is unintended."
            )
        if live > 0 and spec.empty_ok == 1:
            raise Error(
                "empty_ok=1 was declared but the send has "
                + String(live)
                + " live tokens: the declaration no longer matches the case,"
                " which would let a real empty slip through later."
            )

    var p3_recv_ptrs = StaticTuple[
        UnsafePointer[UInt8, MutUntrackedOrigin], P3_MAX_RANKS
    ](UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling())
    var comb_recv_ptrs = Array[
        UnsafePointer[UInt8, MutUntrackedOrigin], ep_n_ranks
    ](fill=UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling())
    var comb_count_ptrs = Array[
        UnsafePointer[UInt64, MutUntrackedOrigin], ep_n_ranks
    ](fill=UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling())
    for rk in range(ep_n_ranks):
        p3_recv_ptrs[rk] = (
            recv_fused[rk].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        )
        comb_recv_ptrs[rk] = (
            recv_unfused[rk]
            .unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        comb_count_ptrs[rk] = (
            recv_count[rk].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        )

    var ep_counters = EPLocalSyncCounters[ep_n_experts](
        counters_dev.unsafe_ptr()
    )
    var arrival_ptr = arrival_dev.unsafe_ptr().as_unsafe_any_origin()

    # ---- fused: one launch ----
    if spec.arm == ARM_SEND:
        mega_ffn_swiglu_mxfp8[
            transpose_b=True,
            num_experts=num_experts,
            match_bf16=True,
            use_inplace=True,
            clean_up=POST_SELF_CLEAN_UP,
            mode=MODE_MEGAFFN,
            mma_bn=8,
            cta_group=1,
            clamp_activation=clamp_act,
            p5_direct_scatter=send_compiled,
            p5_row_cache=send_row_cache,
            p4_signal=True,
            # This arm compares the local C against the unfused chain, so the
            # store must stay even though the send is on; `c_store_dead`
            # otherwise follows the send and would elide it.
            c_store_dead=False,
        ](
            c_test_tt,
            o_test_tt,
            s_test_tt,
            a_tt,
            w13_tt,
            w2_tt,
            a_scales_tt,
            w13_scales_tt,
            w2_scales_tt,
            a_offsets_tt,
            a_scale_offsets_tt,
            expert_ids_tt,
            es1_tt,
            es2_tt,
            nae,
            ctx,
            arrival_ptr,
            swiglu_alpha=SWIGLU_ALPHA,
            swiglu_limit=SWIGLU_LIMIT,
            p3_control=0,
            p3_atomic_counter=ep_counters.get_combine_async_ptr(),
            p3_src_info_ptr=src_info_dev.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin](),
            p3_row_base_ptr=row_base_dev.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            p3_recv_buf_ptrs=p3_recv_ptrs,
            p3_n_ranks=ep_n_ranks,
            p3_p2p_world_size=ep_n_ranks,
            p3_top_k=ep_top_k,
            p3_msg_bytes=msg_bytes,
            p3_max_tokens_per_rank=ep_max_tokens_per_rank,
        )
    else:
        mega_ffn_mxfp8_dispatch[
            transpose_b=True,
            num_experts=num_experts,
            mode=MODE_MEGAFFN,
            clamp_activation=clamp_act,
        ](
            c_test_tt,
            o_test_tt,
            s_test_tt,
            a_tt,
            w13_tt,
            w2_tt,
            a_scales_tt,
            w13_scales_tt,
            w2_scales_tt,
            a_offsets_tt,
            a_scale_offsets_tt,
            expert_ids_tt,
            es1_tt,
            es2_tt,
            nae,
            est,
            ctx,
            arrival_ptr,
            swiglu_alpha=SWIGLU_ALPHA,
            swiglu_limit=SWIGLU_LIMIT,
        )

    # ---- read back ----
    var o_ref_host = alloc[Scalar[a_type]](max(1, o_size))
    var o_test_host = alloc[Scalar[a_type]](max(1, o_size))
    var s_ref_host = alloc[Scalar[sf_type]](max(1, s_total))
    var s_test_host = alloc[Scalar[sf_type]](max(1, s_total))
    var c_ref_host = alloc[Scalar[c_type]](max(1, c_size))
    var c_test_host = alloc[Scalar[c_type]](max(1, c_size))
    var ac_host = alloc[Scalar[DType.uint32]](ac_len)
    ctx.enqueue_copy(o_ref_host, o_ref_dev)
    ctx.enqueue_copy(o_test_host, o_test_dev)
    ctx.enqueue_copy(s_ref_host, s_ref_dev)
    ctx.enqueue_copy(s_test_host, s_test_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.enqueue_copy(c_test_host, c_test_dev)
    ctx.enqueue_copy(ac_host, arrival_dev)
    ctx.synchronize()

    comptime if POISON == 2:
        if c_size > 0:
            # An exponent bit, not an additive nudge: adding 1.0 to a bf16 of
            # magnitude 1e3 is absorbed by the rounding and changes no byte, so
            # an additive poison passes and proves nothing.
            var cb = c_test_host.bitcast[UInt8]()
            var bi = (c_size // 2) * 2 + 1
            cb[bi] = cb[bi] ^ UInt8(0x08)

    # ---- leg 1: fused vs unfused, byte-exact ----
    _compare_bytes["fused_vs_unfused_O"](
        o_ref_host.bitcast[UInt8](), o_test_host.bitcast[UInt8](), o_size
    )
    _compare_bytes["fused_vs_unfused_S"](
        s_ref_host.bitcast[UInt8](), s_test_host.bitcast[UInt8](), s_total
    )
    _compare_bytes["fused_vs_unfused_C"](
        c_ref_host.bitcast[UInt8](),
        c_test_host.bitcast[UInt8](),
        c_size * size_of[Scalar[c_type]](),
    )
    # POST_SELF_CLEAN_UP: this launch's arrivals belong in the half its
    # generation parity selects and nowhere else, so residue in the idle half
    # -- or a generation that never advanced -- would corrupt the next launch
    # on the same buffer. Both geometry-dependent expectations are derived
    # from the buffer rather than pinned: `_selected_tile_height` mirrors the
    # dispatch cascade for the case log, and asserting against a mirror would
    # fail on the mirror rather than on the kernel.
    # A shape with no tiles at all is dropped by the launcher before it
    # builds a descriptor, so it never reaches the kernel and never flips the
    # generation; generation 0 is then itself the assertion, and the scan
    # requires a pristine buffer.
    _ = assert_arrival_slots_after_launch[
        POST_SELF_CLEAN_UP,
        fail_marker="FUZZ_CONTRACT_FAIL kind=arrival_state",
    ](
        ac_host,
        m_blocks + 1,
        "arrival state after the fused launch",
        arrived_pools=ARRIVAL_UNKNOWN,
        l1_arrivals=ARRIVAL_UNKNOWN,
        expect_gen=1 if m_blocks > 0 else 0,
    )

    if probe_out:
        var probe_dst = probe_out.value()
        for i in range(probe_m * N2):
            probe_dst[i] = c_test_host[i]

    # ---- legs 2 and 3: the independent FP64 reference ----
    if check and total > 0:
        if total > REF_MAX_TOKENS:
            print(
                "  FUZZ_NOTE indep_ref_skipped total=",
                total,
                " > ",
                REF_MAX_TOKENS,
                (
                    " (the O(M*N1*K1) host recompute would dominate the"
                    " per-case timeout and a timeout reads as a kernel hang)"
                ),
                sep="",
            )
        else:
            var c_exp = alloc[Float64](c_size)
            var sq_k = alloc[Float32](max(1, total * k2_blocks))
            var s_test_view = TileTensor(s_test_host, s_shape)
            for e in range(nae):
                for local in range(counts[e]):
                    var m = offs[e] + local
                    for hb in range(k2_blocks):
                        sq_k[m * k2_blocks + hb] = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VEC
                        ](s_test_view, sf_row[e] + local, hb * SF_VEC).cast[
                            DType.float32
                        ]()
            _ref_l2(
                o_test_host,
                sq_k,
                w2_host,
                w2_sf_plain,
                es2_host,
                counts,
                ids,
                offs,
                c_exp,
            )
            var acc = Float64(0)
            for i in range(c_size):
                acc += c_exp[i] * c_exp[i]
            var scale = sqrt(acc / Float64(max(1, c_size)))
            # fp32 sequential accumulation over K2 terms, worst case
            # `K2 * 2^-24 * sum|term|`, with `sum|term| ~ sqrt(K2) * rms(|C|)`
            # for a random-sign dot product; doubled for slack.
            var atol_acc = (
                scale * Float64(K2) * sqrt(Float64(K2)) * Float64(1.2e-7)
            )
            # One bf16 rounding of the output. Two roundings of the same real
            # number differ by at most one ULP; against the exact value the
            # budget is half that, so this is already slack.
            comptime BF16_RTOL = Float64(1.05) / Float64(256)
            _tol_check["indep_vs_fused_L2_only"](
                c_test_host, c_exp, c_size, atol_acc, BF16_RTOL, 0.0
            )

            var u_ref = alloc[Float64](max(1, o_size))
            for i in range(o_size):
                u_ref[i] = Float64(0)
            var oq_ref = alloc[Scalar[a_type]](max(1, o_size))
            var sq_ref = alloc[Float32](max(1, total * k2_blocks))
            var c_exp_e2e = alloc[Float64](c_size)
            _ref_l1(
                a_host,
                a_sf_plain,
                w13_host,
                w13_sf_plain,
                es1_host,
                counts,
                ids,
                offs,
                sf_row,
                clamp_act,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                u_ref,
            )
            _ref_quant(u_ref, total, oq_ref, sq_ref)
            # The intermediate leg, reported on its own: the fused L1 and the
            # independent L1 must agree to within the e4m3 rounding of the
            # requantize. A 1-ULP flip is a legitimate per-element event, so
            # the gate is a bounded FRACTION of differing elements; a real L1
            # defect moves essentially all of them.
            var l1_diff = 0
            var l1_max_rel = Float64(0)
            # Compared as raw bytes: an `!=` between two runtime-valued
            # `Scalar[float8_e4m3fn]` crashes the compiler in
            # `FCmpInst::AssertOK`, and the byte compare is what "same
            # quantized element" means anyway.
            var o_k_bytes = o_test_host.bitcast[UInt8]()
            var o_r_bytes = oq_ref.bitcast[UInt8]()
            for i in range(o_size):
                var kb = i % H // SF_VEC
                var m = i // H
                var dq_k = (
                    o_test_host[i].cast[DType.float64]()
                    * sq_k[m * k2_blocks + kb].cast[DType.float64]()
                )
                var dq_r = (
                    oq_ref[i].cast[DType.float64]()
                    * sq_ref[m * k2_blocks + kb].cast[DType.float64]()
                )
                if (
                    o_k_bytes[i] != o_r_bytes[i]
                    or sq_k[m * k2_blocks + kb] != sq_ref[m * k2_blocks + kb]
                ):
                    l1_diff += 1
                    var rel = abs(dq_k - dq_r) / (abs(dq_r) + 1e-300)
                    if rel > l1_max_rel:
                        l1_max_rel = rel
            print(
                "FUZZ_ERR name=indep_vs_fused_L1_intermediate n=",
                o_size,
                " n_bad=",
                l1_diff,
                " frac_bad=",
                Float64(l1_diff) / Float64(max(1, o_size)),
                " max_rel=",
                l1_max_rel,
                sep="",
            )
            if Float64(l1_diff) > 1e-2 * Float64(o_size):
                print(
                    (
                        "FUZZ_NUMERIC_FAIL kind=tolerance"
                        " name=indep_vs_fused_L1_intermediate n_bad="
                    ),
                    l1_diff,
                    sep="",
                )
                raise Error(
                    "independent-reference mismatch: the fused L1"
                    " SwiGLU+requantize disagrees on more than 1% of the"
                    " intermediate"
                )
            _ref_l2(
                oq_ref,
                sq_ref,
                w2_host,
                w2_sf_plain,
                es2_host,
                counts,
                ids,
                offs,
                c_exp_e2e,
            )
            # Adds a possible single-element e4m3 flip in the intermediate
            # (relative step 2^-3), whose contribution to C is one of K2 terms:
            # `2^-3 * max|term| ~ 2^-3 * 4 * rms(|C|)/sqrt(K2)`.
            var atol_e2e = atol_acc + scale * Float64(0.5) / sqrt(Float64(K2))
            _tol_check["indep_vs_fused_end_to_end"](
                c_test_host, c_exp_e2e, c_size, atol_e2e, BF16_RTOL, 5e-3
            )
            _tol_check["indep_vs_unfused_end_to_end"](
                c_ref_host, c_exp_e2e, c_size, atol_e2e, BF16_RTOL, 5e-3
            )
            c_exp.free()
            sq_k.free()
            u_ref.free()
            oq_ref.free()
            sq_ref.free()
            c_exp_e2e.free()

    # ---- send arm: fused vs unfused vs an independent host scatter ----
    if spec.arm == ARM_SEND:
        var exp_bytes = List[UnsafePointer[UInt8, MutUntrackedOrigin]]()
        var got_fused = List[UnsafePointer[UInt8, MutUntrackedOrigin]]()
        var got_unfused = List[UnsafePointer[UInt8, MutUntrackedOrigin]]()
        for _ in range(ep_n_ranks):
            exp_bytes.append(alloc[UInt8](recv_buf_slots * msg_bytes))
            got_fused.append(alloc[UInt8](recv_buf_slots * msg_bytes))
            got_unfused.append(alloc[UInt8](recv_buf_slots * msg_bytes))
        for rk in range(ep_n_ranks):
            for i in range(recv_buf_slots * msg_bytes):
                exp_bytes[rk][i] = SENTINEL

        # The independent scatter: the documented destination formula applied
        # to the kernel's OWN output rows. No kernel code, no shared helper.
        var c_bytes = c_test_host.bitcast[UInt8]()
        var n_written = 0
        for e in range(nae):
            if ids[e] < 0:
                continue
            var cur = offs[e]
            for rk in range(ep_n_ranks):
                var cnt = rank_counts[e * ep_n_ranks + rk]
                for t in range(cnt):
                    var row = cur + t
                    var si = Int(src_info_host[row * 2])
                    var stk = Int(src_info_host[row * 2 + 1])
                    var slot = si * ep_top_k + stk
                    for b in range(msg_bytes):
                        exp_bytes[rk][slot * msg_bytes + b] = c_bytes[
                            row * msg_bytes + b
                        ]
                    n_written += 1
                cur += cnt
        print("    send expected slots written=", n_written, sep="")

        for rk in range(ep_n_ranks):
            ctx.enqueue_copy(got_fused[rk], recv_fused[rk])
        ctx.synchronize()

        comptime if send_compiled:
            comptime assert send_row_cache, (
                "the in-epilogue send's per-row destination cache must be"
                " compiled in with it: without `-D P5_ROW_CACHE=true` the"
                " cache pointer this target supplies is never consulted, and"
                " the arm silently measures a different resolve path"
            )
            for rk in range(ep_n_ranks):
                _compare_bytes["fused_send_vs_independent"](
                    exp_bytes[rk], got_fused[rk], recv_buf_slots * msg_bytes
                )
        else:
            var touched = 0
            for rk in range(ep_n_ranks):
                for i in range(recv_buf_slots * msg_bytes):
                    if got_fused[rk][i] != SENTINEL:
                        touched += 1
            print(
                "    FUZZ_NOTE fused_send_not_compiled touched_bytes=",
                touched,
                (
                    " (build with `-D P5_DIRECT_SCATTER=true -D"
                    " P5_ROW_CACHE=true` to run this leg; the count above is"
                    " the complementary check that nothing wrote the receive"
                    " buffer when the mechanism is out)"
                ),
                sep="",
            )
            if touched != 0:
                print(
                    (
                        "FUZZ_CONTRACT_FAIL kind=send_writes_without_mechanism"
                        " touched="
                    ),
                    touched,
                    sep="",
                )
                raise Error(
                    "the receive buffer was written by a build with no send"
                    " compiled in"
                )

        # Unfused: the standalone combine send, same payload, same tables.
        comptime hw = type_of(ctx).default_device_info
        comptime combine_async = combine_async_kernel[
            c_type,
            hw.max_thread_block_size,
            type_of(c_test_tt.layout),
            type_of(row_major((Idx[recv_capacity_rows], Idx[2]))),
            hw.sm_count,
            ep_top_k,
            ep_n_experts,
            ep_n_ranks,
            msg_bytes,
            ep_max_tokens_per_rank,
            ep_n_ranks,
            use_shmem=False,
        ]
        var src_info_tt = TileTensor(
            src_info_dev, row_major((Idx[recv_capacity_rows], Idx[2]))
        )
        ctx.enqueue_function[combine_async](
            c_test_tt.as_immut(),
            src_info_tt.as_immut(),
            send_stage_dev.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ](),
            comb_recv_ptrs,
            comb_count_ptrs,
            ep_counters,
            Int32(0),
            grid_dim=hw.sm_count,
            block_dim=hw.max_thread_block_size,
        )
        ctx.synchronize()
        for rk in range(ep_n_ranks):
            ctx.enqueue_copy(got_unfused[rk], recv_unfused[rk])
        ctx.synchronize()
        for rk in range(ep_n_ranks):
            _compare_bytes["unfused_send_vs_independent"](
                exp_bytes[rk], got_unfused[rk], recv_buf_slots * msg_bytes
            )

        # The token ranges are SINGLE-USE: the standalone combine above has
        # consumed them (it zeroes each per-(expert, rank) entry as it drains
        # it), so anything that resolves through them again must re-publish
        # first. Printed unconditionally as the standing witness of the
        # consume.
        print(
            "    send live tokens after the standalone combine=",
            _send_live_tokens(ctx, counters_dev),
            sep="",
        )

        # ---- C-elision arm: the send is C's ONLY consumer ----
        #
        # `c_store_dead` drops the local output store and its TMA encode, so C
        # is handed to the kernel as a ZERO-ELEMENT allocation. Three things
        # are under test at once: that the elided path still sends
        # byte-identical rows, that no descriptor is built over a zero global
        # extent (a tensor map rejects `globalDim == 0` outright, while
        # accepting `globalDim < boxDim`, so a surviving encode fails the
        # launch rather than passing against a real allocation), and that a
        # genuinely unbacked C base pointer is never dereferenced.
        comptime if send_compiled:
            if spec.c_dead == 1:
                ctx.enqueue_copy(counters_dev, counters_host)
                ctx.enqueue_memset(row_base_dev, UInt64(0))
                ctx.enqueue_memset(arrival_dev, UInt32(0))
                for rk in range(ep_n_ranks):
                    ctx.enqueue_memset(recv_fused[rk], SENTINEL)
                ctx.synchronize()
                var live2 = _send_live_tokens(ctx, counters_dev)
                print(
                    "    c_dead: live tokens after re-publish=", live2, sep=""
                )
                if live2 <= 0 and spec.empty_ok == 0:
                    raise Error(
                        "VACUOUS GATE: the re-published token ranges are"
                        " empty, so the C-elision send would move nothing and"
                        " the comparison after it would pass on a buffer"
                        " nobody wrote."
                    )
                var c_zero_dev = ctx.enqueue_create_buffer[c_type](0)
                var c_zero_tt = TileTensor(
                    c_zero_dev, row_major(Coord(Int(0), Idx[N2]))
                )
                # A zero-element allocation succeeds and hands back an
                # UNBACKED address, so the pointer value is worth printing:
                # it is what the kernel would fault on if any store survived.
                print(
                    "    c_dead: C elements=0 base_ptr=",
                    Int(c_zero_dev.unsafe_ptr()),
                    " (unbacked)",
                    sep="",
                )
                mega_ffn_swiglu_mxfp8[
                    transpose_b=True,
                    num_experts=num_experts,
                    match_bf16=True,
                    use_inplace=True,
                    clean_up=POST_SELF_CLEAN_UP,
                    mode=MODE_MEGAFFN,
                    mma_bn=8,
                    cta_group=1,
                    clamp_activation=clamp_act,
                    p5_direct_scatter=send_compiled,
                    p5_row_cache=send_row_cache,
                    p4_signal=True,
                    c_store_dead=True,
                ](
                    c_zero_tt,
                    o_test_tt,
                    s_test_tt,
                    a_tt,
                    w13_tt,
                    w2_tt,
                    a_scales_tt,
                    w13_scales_tt,
                    w2_scales_tt,
                    a_offsets_tt,
                    a_scale_offsets_tt,
                    expert_ids_tt,
                    es1_tt,
                    es2_tt,
                    nae,
                    ctx,
                    arrival_ptr,
                    swiglu_alpha=SWIGLU_ALPHA,
                    swiglu_limit=SWIGLU_LIMIT,
                    p3_control=0,
                    p3_atomic_counter=ep_counters.get_combine_async_ptr(),
                    p3_src_info_ptr=src_info_dev.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmUntrackedOrigin](),
                    p3_row_base_ptr=row_base_dev.unsafe_ptr().unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                    p3_recv_buf_ptrs=p3_recv_ptrs,
                    p3_n_ranks=ep_n_ranks,
                    p3_p2p_world_size=ep_n_ranks,
                    p3_top_k=ep_top_k,
                    p3_msg_bytes=msg_bytes,
                    p3_max_tokens_per_rank=ep_max_tokens_per_rank,
                )
                ctx.synchronize()
                for rk in range(ep_n_ranks):
                    ctx.enqueue_copy(got_fused[rk], recv_fused[rk])
                ctx.synchronize()
                # At zero tokens `exp_bytes` is entirely sentinel, so the same
                # comparison asserts the complementary invariant: an elided-C
                # launch with nothing routed must write nothing anywhere.
                for rk in range(ep_n_ranks):
                    _compare_bytes["c_dead_send_vs_independent"](
                        exp_bytes[rk],
                        got_fused[rk],
                        recv_buf_slots * msg_bytes,
                    )
                _ = c_zero_dev^
        else:
            if spec.c_dead == 1:
                print(
                    "    FUZZ_NOTE c_dead_arm_not_compiled -- the C-elision"
                    " path needs `-D P5_DIRECT_SCATTER=true -D"
                    " P5_ROW_CACHE=true`, because the epilogue asserts that"
                    " eliding the local store requires the peer send to be"
                    " compiled in"
                )

        for rk in range(ep_n_ranks):
            exp_bytes[rk].free()
            got_fused[rk].free()
            got_unfused[rk].free()

    # ---- determinism: re-launch the same input and require bit-stability ----
    if rerun > 0 and spec.arm == ARM_FFN and total > 0:
        var first = alloc[Scalar[c_type]](c_size)
        for i in range(c_size):
            first[i] = c_test_host[i]
        for _ in range(rerun - 1):
            ctx.enqueue_memset(c_test_dev, Scalar[c_type](-777.25))
            ctx.enqueue_memset(arrival_dev, UInt32(0))
            mega_ffn_mxfp8_dispatch[
                transpose_b=True,
                num_experts=num_experts,
                mode=MODE_MEGAFFN,
                clamp_activation=clamp_act,
            ](
                c_test_tt,
                o_test_tt,
                s_test_tt,
                a_tt,
                w13_tt,
                w2_tt,
                a_scales_tt,
                w13_scales_tt,
                w2_scales_tt,
                a_offsets_tt,
                a_scale_offsets_tt,
                expert_ids_tt,
                es1_tt,
                es2_tt,
                nae,
                est,
                ctx,
                arrival_ptr,
                swiglu_alpha=SWIGLU_ALPHA,
                swiglu_limit=SWIGLU_LIMIT,
            )
            ctx.enqueue_copy(c_test_host, c_test_dev)
            ctx.synchronize()
            _compare_bytes["rerun_determinism_C"](
                first.bitcast[UInt8](),
                c_test_host.bitcast[UInt8](),
                c_size * size_of[Scalar[c_type]](),
            )
        first.free()

    a_offsets_host.free()
    a_scale_offsets_host.free()
    expert_ids_host.free()
    w13_host.free()
    w2_host.free()
    w13_sf_plain.free()
    w2_sf_plain.free()
    w13_scales_host.free()
    w2_scales_host.free()
    es1_host.free()
    es2_host.free()
    a_host.free()
    a_sf_plain.free()
    a_scales_host.free()
    counters_host.free()
    src_info_host.free()
    o_ref_host.free()
    o_test_host.free()
    s_ref_host.free()
    s_test_host.free()
    c_ref_host.free()
    c_test_host.free()
    ac_host.free()
    _ = a_dev^
    _ = a_scales_dev^
    _ = w13_dev^
    _ = w13_scales_dev^
    _ = w2_dev^
    _ = w2_scales_dev^
    _ = es1_dev^
    _ = es2_dev^
    _ = a_offsets_dev^
    _ = a_scale_offsets_dev^
    _ = expert_ids_dev^
    _ = o_ref_dev^
    _ = o_test_dev^
    _ = s_ref_dev^
    _ = s_test_dev^
    _ = c_ref_dev^
    _ = c_test_dev^
    _ = arrival_dev^
    _ = counters_dev^
    _ = src_info_dev^
    _ = row_base_dev^
    _ = send_stage_dev^
    _ = recv_fused^
    _ = recv_unfused^
    _ = recv_count^


# ===----------------------------------------------------------------------=== #
# Batch invariance
# ===----------------------------------------------------------------------=== #
#
# A token's FFN output must not depend on what it is co-batched with. This is a
# real hazard for a MoE FFN: the tile geometry comes from an `avg_m` cascade
# keyed on `estimated_total_m / num_active_experts`, so co-batched tokens can
# change `mma_bn` under a token that did not move.
#
# Both compositions are kept inside ONE regime, so the gate is honestly
# bit-exact. Measured on this kernel the probe's rows are in fact bit-identical
# even ACROSS the regime boundary -- the cascade re-tiles the TOKEN and OUTPUT
# axes, not the K reduction, so a token's own dot-product order does not change
# with `mma_bn`. A cross-regime divergence control therefore has no teeth here
# and is not provided; the oracle's teeth come from `-D mff_poison=7`, which
# perturbs the probe's own token values in one composition and must make it
# fail.


def _probe_pair(
    ctx: DeviceContext,
    spec: CaseSpec,
    probe_m: Int,
    n_fill_a: Int,
    filler_m_a: Int,
    n_fill_b: Int,
    filler_m_b: Int,
) raises -> Tuple[Int, Int]:
    """Runs the probe under two compositions; returns (n_diff_bytes, n_bytes).
    """
    var n = probe_m * N2
    var pa = alloc[Scalar[c_type]](max(1, n))
    var pb = alloc[Scalar[c_type]](max(1, n))
    run_case(
        ctx,
        spec,
        probe_m=probe_m,
        n_fillers=n_fill_a,
        filler_m=filler_m_a,
        filler_seed=spec.tok_seed ^ 0x11,
        probe_out=pa,
    )
    run_case(
        ctx,
        spec,
        probe_m=probe_m,
        n_fillers=n_fill_b,
        filler_m=filler_m_b,
        filler_seed=spec.tok_seed ^ 0x9E37_79B9,
        probe_out=pb,
        probe_seed_bump=1 if POISON == 7 else 0,
    )
    var a8 = pa.bitcast[UInt8]()
    var b8 = pb.bitcast[UInt8]()
    var nb = n * size_of[Scalar[c_type]]()
    var diff = 0
    for i in range(nb):
        if a8[i] != b8[i]:
            diff += 1
    pa.free()
    pb.free()
    return (diff, nb)


def _fill_sizes_same_regime(probe_m: Int) -> Tuple[Int, Int, Int, Int]:
    """Two filler shapes whose `avg_m` both land in the probe's regime, with
    DIFFERENT filler counts (so the launch grid differs) and totals."""
    var uppers = _regime_upper()
    var r = 0
    while r < len(uppers) - 1 and probe_m > uppers[r]:
        r += 1
    var hi = uppers[r] if uppers[r] > 0 else probe_m
    var lo = 1 if r == 0 else uppers[r - 1] + 1
    var f = max(lo, min(hi, probe_m))
    var na = max(1, min(2, num_experts - 1))
    var nb = max(1, min(num_experts - 1, na + 2))
    return (na, f, nb, f)


def run_batch_invariance_case(ctx: DeviceContext, spec: CaseSpec) raises:
    seed(spec.tok_seed)
    var probe_m = boundary_int(1, 64, 8)
    var sizes = _fill_sizes_same_regime(probe_m)
    var out = _probe_pair(
        ctx, spec, probe_m, sizes[0], sizes[1], sizes[2], sizes[3]
    )
    print(
        "    batch_invariance probe_m=",
        probe_m,
        " fillers=",
        sizes[0],
        "x",
        sizes[1],
        " vs ",
        sizes[2],
        "x",
        sizes[3],
        " diff_bytes=",
        out[0],
        "/",
        out[1],
        sep="",
    )
    if out[0] != 0:
        print(
            "FUZZ_NUMERIC_FAIL kind=batch_invariance n_bad=",
            out[0],
            " n=",
            out[1],
            sep="",
        )
        raise Error(
            "MegaFFN is not batch-invariant within one avg_m regime: the"
            " probe's output changed with the co-batch composition"
        )


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    var batch_invariance = flag_int(args, "--batch-invariance", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "arm=",
                specs[i].arm,
                "shape_class=",
                specs[i].shape_class,
                "num_active_experts=",
                specs[i].num_active_experts,
                "tok_seed=",
                specs[i].tok_seed,
                "empty_ok=",
                specs[i].empty_ok,
                "c_dead=",
                specs[i].c_dead,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--arm", ARM_FFN),
            flag_int(args, "--shape_class", SC_RANDOM),
            flag_int(args, "--num_active_experts", 2),
            flag_int(args, "--tok_seed", 1),
            flag_int(args, "--empty_ok", 0),
            flag_int(args, "--c_dead", 0),
        )
        print("FUZZ_SINGLE ", spec, sep="")
        with DeviceContext() as ctx:
            if batch_invariance:
                run_batch_invariance_case(ctx, spec)
            elif rerun > 0:
                run_case(ctx, spec, rerun=rerun)
            else:
                run_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_mega_ffn seed=",
        the_seed,
        " budget=",
        the_budget,
        " E=",
        num_experts,
        " N1=",
        N1,
        " K1=",
        K1,
        " N2=",
        N2,
        " ranks=",
        ep_n_ranks,
        " top_k=",
        ep_top_k,
        " send_compiled=",
        send_compiled,
        " ===",
        sep="",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case ", i, ": ", specs[i], sep="")
            if batch_invariance:
                run_batch_invariance_case(ctx, specs[i])
            elif rerun > 0:
                run_case(ctx, specs[i], rerun=rerun)
            else:
                run_case(ctx, specs[i], check)
    print("=== done: ", len(specs), " cases ===", sep="")
