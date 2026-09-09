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
# Fuzz target: `block_select_topk` -- the MiniMax-M3 sparse-attention (MSA)
# indexer's block-selection core (see gpu-kernels-fuzzing-design.md).
#
# `block_select_topk` (max/kernels/src/nn/attention/gpu/sparse_indexer_common.mojo)
# is the cooperative, single-CTA-per-row top-k that BOTH the M3 indexer phases
# share: the prefill scorer (sparse_indexer_prefill) and the decode scorer
# (sparse_indexer_decode) each produce a row of per-block QK scores, then call
# this primitive to select the top-k block indices to attend to. It is purely a
# selection step (value -> index), so it is hardware-independent (no SM100 /
# tensor-core machinery) and runs on any non-Apple GPU, which makes it a clean,
# portable fuzz target with a strong reference.
#
# A test launcher (`_select_test_kernel`, copied from the kernel's own unit test
# test_sparse_indexer_common.mojo) runs one thread block per score row and calls
# `block_select_topk` on that row. We fuzz the runtime shape (num_rows,
# num_blocks, k), the launch width (block_dim, a warp multiple), and the score
# value distribution (uniform/normal/sparse/large/all-equal/specials, plus the
# indexer-specific all-dead and boundary-tie rows).
#
# Oracles (a selection kernel has both a memory-safety and a correctness face):
#
#  - VALIDITY CONTRACT (always on, so it rides under diff/memcheck/redzone/
#    poison too). The fresh int32 output buffer is poison-filled with -2; the
#    leading score element is an OOB pre-guard canary. For each row we recompute
#    `n_written = min(min(k, num_blocks), n_selectable)` on the host and check:
#    every output in [0, n_written) is a distinct, in-range index pointing at a
#    SELECTABLE block (not -inf / not NaN); the tail [n_written, k) is all -1; a
#    no-winner row (all -inf / all NaN) is all -1; and the OOB guard is intact.
#    A garbage / negative / sentinel(-2) / dead-block / out-of-range index FAILs
#    here even with no sanitizer (FUZZ_CONTRACT_FAIL).
#
#  - `ref` (--check 1): the top-k correctness invariant -- every selected score
#    >= every non-selected SELECTABLE score (tie-robust, so legitimate ties do
#    not cry wolf). This subsumes the forced-1e30-outlier guarantee while still
#    allowing a forced block to be outranked by >= k blocks at +inf. Emits
#    FUZZ_NUMERIC_FAIL.
#
#  - `contract` (--contract 1): force NaN/Inf/+-0/+-max into every row and assert
#    the same validity contract holds (no out-of-range / dead-block selection).
#
# `--inject N` (default 0; never set by the orchestrator) is the oracle canary:
#   1 = report a valid-but-wrong selection (an excluded low block) on row 0 ->
#       trips `ref` (invariant) but PASSES the validity contract, proving ref
#       catches what diff/contract miss;
#   2 = write an out-of-range index on row 0 -> trips the validity contract
#       even under diff.
#
# Spec fields (all Int, so the orchestrator generates/shrinks generically):
#   num_rows, num_blocks, k, block_dim, dist, force (0/1), seed.

from std.collections import Set
from max.gpu import block_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import max, min
from std.random import rand, random_ui64, seed as set_seed
from std.sys.defines import get_defined_int
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf

from layout import TensorLayout, TileTensor, row_major

from nn.attention.gpu.sparse_indexer_common import block_select_topk

from _fuzz import (
    VD_ALL_EQUAL,
    VD_LARGE,
    VD_NORMAL,
    VD_SPARSE,
    VD_SPECIALS,
    VD_UNIFORM,
    boundary_int,
    collect_args,
    fill_by_dist,
    fill_with_specials,
    flag,
    flag_int,
)

comptime score_type = DType.float32
comptime out_idx_type = DType.int32
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# block_dim is the strided-scan width; num_blocks crossing it (and the warp size
# 32 nested inside `_block_reduce_topk`) are the dispatch moduli. 128 is the
# canonical width the kernel's own test uses, so it is the boundary tile here.
comptime TILE_BLOCKS = 128
# M3 selects topk=16 blocks; the FP8 indexer fill path mentions up to 2048. 16
# is the realistic dispatch modulus, so most k draws land small and cheap.
comptime TILE_K = 16

# Value-distribution ids. 0..5 reuse the shared _fuzz VD_* fills; 6/7 are the
# indexer-specific degenerate rows the kernel's own test exercises.
comptime IDX_ALL_DEAD = 6  # every score -inf (the kernel's dead sentinel)
comptime IDX_BOUNDARY_TIE = 7  # distinct highs + a tie for the last slot
comptime NUM_DISTS = 8


def _dist_name(d: Int) -> String:
    if d == VD_NORMAL:
        return "normal"
    if d == VD_SPARSE:
        return "sparse"
    if d == VD_LARGE:
        return "large"
    if d == VD_ALL_EQUAL:
        return "all_equal"
    if d == VD_SPECIALS:
        return "specials"
    if d == IDX_ALL_DEAD:
        return "all_dead"
    if d == IDX_BOUNDARY_TIE:
        return "boundary_tie"
    return "uniform"


def _is_selectable(v: Float32) -> Bool:
    """True iff `v` can win a slot (finite or +inf; never NaN or the -inf dead
    sentinel) -- matches `block_select_topk`'s `elem > dead_val` selection rule.
    """
    if v != v:  # NaN never beats anything
        return False
    return v != min_or_neg_inf[score_type]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var num_rows: Int  # score rows (grid_dim.x); one CTA selects per row
    var num_blocks: Int  # block scores per row (the key/context axis)
    var k: Int  # number of indices to emit per row (top-k)
    var block_dim: Int  # launch width (a multiple of the warp size)
    var dist: Int  # value-distribution id (see _dist_name)
    var force: Int  # 0/1: pin one block to 1e30 (it must be selected)
    var seed: Int  # drives the value fill + the forced index

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "num_rows=",
            self.num_rows,
            " num_blocks=",
            self.num_blocks,
            " k=",
            self.k,
            " block_dim=",
            self.block_dim,
            " dist=",
            _dist_name(self.dist),
            " force=",
            self.force,
            " seed=",
            self.seed,
        )


def _pick_block_dim() -> Int:
    """A warp-multiple launch width, biased to 128 (the kernel's canonical CTA).

    `block_select_topk` requires `block_dim.x` to be a multiple of the warp size
    (32 on NVIDIA, 64 on AMD); 64 is safe on both. All draws are run through
    `_snap_block_dim` at launch, so the generic orchestrator shrink reducing this
    field to a non-warp-multiple (e.g. 1) can never produce an invalid geometry.
    """
    var roll = Int(random_ui64(0, 7))
    if roll <= 3:
        return 128
    if roll == 4:
        return 64
    if roll == 5:
        return 256
    if roll == 6:
        return 512
    return 1024


def _snap_block_dim(bd: Int) -> Int:
    """Round `bd` up to a valid launch width: a multiple of 64 in [64, 1024].

    64 is a multiple of both the NVIDIA (32) and AMD (64) warp sizes, so a snapped
    width satisfies `block_select_topk`'s precondition on either vendor. This
    keeps every launch valid regardless of what the orchestrator's generic shrink
    passes for `block_dim` (it treats the field as a free Int).
    """
    var w = 64
    var up = ((max(1, bd) + w - 1) // w) * w
    return max(64, min(1024, up))


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var num_rows = boundary_int(1, 64, 8)
        var num_blocks = boundary_int(1, 8192, TILE_BLOCKS)
        var k = boundary_int(1, 2048, TILE_K)
        var block_dim = _pick_block_dim()
        # Bias to tie-free uniform (the exact-rank regime), but sample every
        # degenerate distribution: all-equal stresses winner eviction, specials
        # / all-dead drive the no-winner and partial-winner paths, boundary-tie
        # stresses the cut line.
        var dist: Int
        var droll = Int(random_ui64(0, 9))
        if droll <= 3:
            dist = VD_UNIFORM
        elif droll == 4:
            dist = VD_NORMAL
        elif droll == 5:
            dist = VD_SPARSE
        elif droll == 6:
            dist = VD_ALL_EQUAL
        elif droll == 7:
            dist = VD_SPECIALS
        elif droll == 8:
            dist = IDX_ALL_DEAD
        else:
            dist = IDX_BOUNDARY_TIE
        var force = Int(random_ui64(0, 2) == 0)  # ~1/3 of cases force a block
        var the_seed = Int(random_ui64(1, 1_000_000))
        specs.append(
            CaseSpec(num_rows, num_blocks, k, block_dim, dist, force, the_seed)
        )
    return specs^


# ===----------------------------------------------------------------------=== #
# Launcher kernel (one thread block per row) -- copied from the kernel's own
# unit test (test_sparse_indexer_common.mojo:_select_test_kernel).
# ===----------------------------------------------------------------------=== #


def _select_test_kernel[
    ScoresLT: TensorLayout,
    OutLT: TensorLayout,
](
    scores: TileTensor[score_type, ScoresLT, MutAnyOrigin],
    out_idxs: TileTensor[out_idx_type, OutLT, MutAnyOrigin],
    num_blocks_dev: Int32,
    k_dev: Int32,
):
    comptime assert scores.flat_rank == 2 and out_idxs.flat_rank == 2
    # `Int` is not device-passable; widen the fixed-width args.
    var num_blocks = Int(num_blocks_dev)
    var k = Int(k_dev)
    var row = block_idx.x
    var s_lt = scores.to_layout_tensor()
    var o_lt = out_idxs.to_layout_tensor()
    var scores_row = rebind[MutPointer[Scalar[score_type], MutAnyOrigin]](
        s_lt.ptr_at_offset(Index(row, 0))
    )
    var out_row = rebind[MutPointer[Scalar[out_idx_type], MutAnyOrigin]](
        o_lt.ptr_at_offset(Index(row, 0))
    )
    block_select_topk[score_type, out_idx_type](
        scores_row, num_blocks, k, out_row
    )


# ===----------------------------------------------------------------------=== #
# Host-side score fill
# ===----------------------------------------------------------------------=== #


def _fill_scores(
    scores_host: MutPointer[Scalar[score_type], _],
    num_rows: Int,
    num_blocks: Int,
    k: Int,
    dist: Int,
    contract: Bool,
):
    var n = num_rows * num_blocks
    var span = Span(unsafe_ptr=scores_host, length=n)
    if contract:
        # contract oracle: heavy NaN/Inf/+-0/+-max coverage over every row.
        fill_with_specials(span, density=0.35)
        return
    if dist <= VD_SPECIALS:
        fill_by_dist(span, dist)
        return
    if dist == IDX_ALL_DEAD:
        var dead = min_or_neg_inf[score_type]()
        for i in range(n):
            scores_host[i] = dead
        return
    # IDX_BOUNDARY_TIE: all tied at 50, except the first (k-1) blocks set to
    # distinct clear-highs -- top-k must take {0..k-2} plus one tied block.
    for r in range(num_rows):
        var base = r * num_blocks
        for j in range(num_blocks):
            scores_host[base + j] = Float32(50.0)
        for j in range(min(k - 1, num_blocks)):
            scores_host[base + j] = Float32(1000.0) + Float32(k - 1 - j)


# ===----------------------------------------------------------------------=== #
# One case: fill -> launch -> validity contract (+ optional ref / contract).
# ===----------------------------------------------------------------------=== #


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
    inject: Int = 0,
) raises:
    var num_rows = spec.num_rows
    var num_blocks = spec.num_blocks
    var k = spec.k
    var n_scores = num_rows * num_blocks
    var n_out = num_rows * k

    var dead = min_or_neg_inf[score_type]()

    # --- host scores (the pristine reference; the kernel mutates the device
    # copy in place during extraction, so we never read scores back for ref) ---
    var scores_host = ctx.enqueue_create_host_buffer[score_type](n_scores)
    _fill_scores(
        scores_host.unsafe_ptr(),
        num_rows,
        num_blocks,
        k,
        spec.dist,
        contract,
    )

    # Force a known block to a large value in every row; it must be selected.
    var force_idx = -1
    if spec.force == 1 and num_blocks > 0:
        force_idx = spec.seed % num_blocks
        for r in range(num_rows):
            scores_host[r * num_blocks + force_idx] = Float32(1.0e30)
    ctx.synchronize()

    # Leading guard element at index 0; row data lives at [1, n_scores]. A write
    # to scores[-1] for row 0 (the pre-guard OOB) lands on this canary.
    var canary = Float32(13371337.0)
    var scores_dev = ctx.enqueue_create_buffer[score_type](n_scores + 1)
    var stage = ctx.enqueue_create_host_buffer[score_type](n_scores + 1)
    stage[0] = canary
    for j in range(n_scores):
        stage[1 + j] = scores_host[j]
    ctx.enqueue_copy(dst_buf=scores_dev, src_buf=stage)

    # FRESH output buffer, poison-filled with -2: any unwritten slot survives as
    # -2 and trips the validity contract (it is neither a valid index nor -1).
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](n_out)
    out_dev.enqueue_fill(Int32(-2))

    var data_buf = DeviceBuffer[score_type](
        ctx, scores_dev.unsafe_ptr() + 1, n_scores, owning=False
    )
    var scores_t = TileTensor(data_buf, row_major((num_rows, num_blocks)))
    var out_t = TileTensor(out_dev, row_major((num_rows, k)))

    comptime kernel = _select_test_kernel[
        type_of(scores_t).LayoutType, type_of(out_t).LayoutType
    ]
    ctx.enqueue_function[kernel](
        scores_t,
        out_t,
        Int32(num_blocks),
        Int32(k),
        grid_dim=num_rows,
        block_dim=_snap_block_dim(spec.block_dim),
    )

    var out_host = ctx.enqueue_create_host_buffer[out_idx_type](n_out)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    var full_host = ctx.enqueue_create_host_buffer[score_type](n_scores + 1)
    ctx.enqueue_copy(dst_buf=full_host, src_buf=scores_dev)
    ctx.synchronize()

    # OOB pre-guard: a `scores[-1]` write for row 0 would have stomped the canary.
    if full_host[0] != canary:
        print(
            "FUZZ_CONTRACT_FAIL kind=oob_scores_guard got=",
            full_host[0],
            "expected=",
            canary,
        )
        raise Error("OOB write to scores[-1] before row 0")

    # Oracle canary (never set by the orchestrator): perturb the output so a
    # green oracle is proven able to go FAIL.
    if inject != 0 and num_rows > 0 and k > 0:
        _apply_inject(
            inject,
            out_host.unsafe_ptr(),
            scores_host.unsafe_ptr(),
            num_blocks,
            k,
            force_idx,
            dead,
        )

    # --- VALIDITY CONTRACT (always) -----------------------------------------
    var k_batch = min(k, num_blocks)
    for r in range(num_rows):
        var base = r * num_blocks
        var n_sel = 0
        for j in range(num_blocks):
            if _is_selectable(scores_host[base + j]):
                n_sel += 1
        var n_written = min(k_batch, n_sel)

        var got = Set[Int]()
        for j in range(n_written):
            var idx = Int(out_host[r * k + j])
            if idx < 0 or idx >= num_blocks:
                print(
                    "FUZZ_CONTRACT_FAIL kind=index_out_of_range row=",
                    r,
                    "pos=",
                    j,
                    "idx=",
                    idx,
                    "num_blocks=",
                    num_blocks,
                    "dist=",
                    _dist_name(spec.dist),
                )
                raise Error("selected index out of range [0, num_blocks)")
            if idx in got:
                print(
                    "FUZZ_CONTRACT_FAIL kind=duplicate_index row=",
                    r,
                    "idx=",
                    idx,
                )
                raise Error("duplicate selected index")
            got.add(idx)
            if not _is_selectable(scores_host[base + idx]):
                print(
                    "FUZZ_CONTRACT_FAIL kind=selected_dead_block row=",
                    r,
                    "idx=",
                    idx,
                    "score=",
                    scores_host[base + idx],
                )
                raise Error("selected a dead/NaN block (not selectable)")

        for j in range(n_written, k):
            if Int(out_host[r * k + j]) != -1:
                print(
                    "FUZZ_CONTRACT_FAIL kind=tail_not_sentinel row=",
                    r,
                    "pos=",
                    j,
                    "got=",
                    Int(out_host[r * k + j]),
                    "n_written=",
                    n_written,
                )
                raise Error("output tail must be -1")

        # --- ref invariant (optional) ---------------------------------------
        if check and n_written > 0:
            _check_topk_invariant(
                scores_host.unsafe_ptr(), base, num_blocks, got, r
            )

    _ = scores_dev
    _ = out_dev


def run_schedule_case(ctx: DeviceContext, spec: CaseSpec, repeats: Int) raises:
    """Cross-launch-width determinism oracle (the `schedule` decomposition).

    `block_select_topk` documents a width-independent contract: "ties broken by
    smallest index". So for one fixed score row, the emitted index array must be
    bit-identical no matter how many threads the CTA uses. We run the same
    pristine scores across a set of valid launch widths and flag any divergence
    -- a reduction-order / tie-break leak that diff/ref/memcheck cannot see.

    Each launch needs a fresh device copy of the scores, because the kernel
    evicts winners by overwriting `scores` in place during extraction.
    """
    var num_rows = spec.num_rows
    var num_blocks = spec.num_blocks
    var k = spec.k
    var n_scores = num_rows * num_blocks
    var n_out = num_rows * k

    # Valid widths (multiples of 64; <= 1024). Vary the warp/stride decomposition.
    var widths: List[Int] = [64, 128, 192, 256, 384, 512, 768, 1024]
    var n_variants = min(repeats, len(widths))
    if n_variants < 2:
        n_variants = 2

    # Fill the pristine scores once (mirror run_one_case: dist + optional force).
    var scores_host = ctx.enqueue_create_host_buffer[score_type](n_scores)
    _fill_scores(
        scores_host.unsafe_ptr(), num_rows, num_blocks, k, spec.dist, False
    )
    if spec.force == 1 and num_blocks > 0:
        var force_idx = spec.seed % num_blocks
        for r in range(num_rows):
            scores_host[r * num_blocks + force_idx] = Float32(1.0e30)
    ctx.synchronize()

    var ref_out = ctx.enqueue_create_host_buffer[out_idx_type](n_out)

    for v in range(n_variants):
        var bd = _snap_block_dim(widths[v])

        var scores_dev = ctx.enqueue_create_buffer[score_type](n_scores)
        ctx.enqueue_copy(dst_buf=scores_dev, src_buf=scores_host)
        var out_dev = ctx.enqueue_create_buffer[out_idx_type](n_out)
        out_dev.enqueue_fill(Int32(-2))

        var scores_t = TileTensor(scores_dev, row_major((num_rows, num_blocks)))
        var out_t = TileTensor(out_dev, row_major((num_rows, k)))

        comptime kernel = _select_test_kernel[
            type_of(scores_t).LayoutType, type_of(out_t).LayoutType
        ]
        ctx.enqueue_function[kernel](
            scores_t,
            out_t,
            Int32(num_blocks),
            Int32(k),
            grid_dim=num_rows,
            block_dim=bd,
        )

        var out_host = ctx.enqueue_create_host_buffer[out_idx_type](n_out)
        ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
        ctx.synchronize()

        if v == 0:
            for i in range(n_out):
                ref_out[i] = out_host[i]
        else:
            for i in range(n_out):
                if out_host[i] != ref_out[i]:
                    var row = i // k
                    var pos = i % k
                    print(
                        "FUZZ_NUMERIC_FAIL kind=width_nondeterminism width=",
                        bd,
                        "ref_width=",
                        _snap_block_dim(widths[0]),
                        "row=",
                        row,
                        "pos=",
                        pos,
                        "got=",
                        Int(out_host[i]),
                        "ref=",
                        Int(ref_out[i]),
                        "dist=",
                        _dist_name(spec.dist),
                    )
                    raise Error(
                        "block_select_topk output depends on block_dim"
                        " (tie-break / reduction-order leak)"
                    )
        _ = scores_dev
        _ = out_dev


def _apply_inject(
    inject: Int,
    out_host: MutPointer[Scalar[out_idx_type], _],
    scores_host: MutPointer[Scalar[score_type], _],
    num_blocks: Int,
    k: Int,
    force_idx: Int,
    dead: Float32,
):
    """Corrupt row 0's output to prove an oracle can FAIL (positive control)."""
    if inject == 2:
        # Out-of-range index -> trips the validity contract (even under diff).
        out_host[0] = Scalar[out_idx_type](num_blocks)
        return
    # inject == 1: a valid-but-wrong selection. Find the lowest-scoring
    # selectable block in row 0 that the kernel did NOT place at position 0, and
    # report it as selected. In-range + distinct (so the validity contract still
    # passes), but it violates the top-k invariant -> only `ref` catches it.
    var cur0 = Int(out_host[0])
    var worst_idx = -1
    var worst_val = Float32(0)
    for j in range(num_blocks):
        if j == cur0:
            continue
        var v = scores_host[j]
        if v != v or v == dead:  # not selectable
            continue
        if worst_idx == -1 or v < worst_val:
            worst_val = v
            worst_idx = j
    if worst_idx >= 0:
        out_host[0] = Scalar[out_idx_type](worst_idx)


def _check_topk_invariant(
    scores_host: MutPointer[Scalar[score_type], _],
    base: Int,
    num_blocks: Int,
    got: Set[Int],
    row: Int,
) raises:
    """Top-k correctness: min selected score >= max non-selected SELECTABLE
    score (tie-robust).

    This subsumes the forced-block guarantee: if a high outlier (a forced 1e30
    block, or a +inf special) is wrongly excluded, it becomes a non-selected
    block whose score exceeds some selected score, so `sel_min < non_max` fires.
    A forced block that is *legitimately* outranked (e.g. >= k blocks at +inf,
    which is > 1e30) is correctly NOT required to be selected.
    """
    var sel_min = Float32(0)
    var have_sel = False
    for idx in got:
        var v = scores_host[base + idx]
        if not have_sel or v < sel_min:
            sel_min = v
            have_sel = True

    var non_max = Float32(0)
    var have_non = False
    for i in range(num_blocks):
        if i in got:
            continue
        var v = scores_host[base + i]
        if not _is_selectable(v):
            continue
        if not have_non or v > non_max:
            non_max = v
            have_non = True

    if have_sel and have_non and sel_min < non_max:
        print(
            "FUZZ_NUMERIC_FAIL kind=topk_invariant row=",
            row,
            "sel_min=",
            sel_min,
            "non_max=",
            non_max,
        )
        raise Error("selected element ranked below an excluded one")


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var contract = flag_int(args, "--contract", 0) == 1
    var inject = flag_int(args, "--inject", 0)
    var schedule_repeats = flag_int(args, "--schedule", 0)
    set_seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "num_rows=",
                specs[i].num_rows,
                "num_blocks=",
                specs[i].num_blocks,
                "k=",
                specs[i].k,
                "block_dim=",
                specs[i].block_dim,
                "dist=",
                specs[i].dist,
                "force=",
                specs[i].force,
                "seed=",
                specs[i].seed,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--num_rows", 4),
            flag_int(args, "--num_blocks", 64),
            flag_int(args, "--k", 16),
            flag_int(args, "--block_dim", 128),
            flag_int(args, "--dist", VD_UNIFORM),
            flag_int(args, "--force", 0),
            flag_int(args, "--seed", the_seed),
        )
        print("FUZZ_SINGLE ", spec)
        with DeviceContext() as ctx:
            if schedule_repeats > 0:
                run_schedule_case(ctx, spec, schedule_repeats)
            else:
                run_one_case(ctx, spec, check, contract, inject)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_sparse_indexer seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if schedule_repeats > 0:
                run_schedule_case(ctx, specs[i], schedule_repeats)
            else:
                run_one_case(ctx, specs[i], check, contract, inject)
    print("=== done:", len(specs), "cases ===")
