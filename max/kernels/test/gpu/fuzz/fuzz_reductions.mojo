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
# Fuzz target: the `algorithm.reductions` entry points (`reduce_sum`,
# `reduce_max`, `reduce_min`, `reduce_mean`, `reduce_argmax`, `reduce_argmin`)
# over the `rowwise` GPU scaffolder.
#
# Two things distinguish this target from its normalize-shaped neighbours
# (`fuzz_softmax`, `fuzz_rms_norm`, `fuzz_layer_norm`):
#
# 1. The reduced axis is drawn from `lo=0`, not `lo=1`. A reduction over a
#    zero-extent axis is IN the contract — it owns `product(shape \ axis)`
#    outputs that must receive the monoid identity — so the usual "clamp the
#    generator to lo=1" rule does not apply to this dimension. Getting that
#    wrong is what this target exists to catch: the empty axis used to return
#    without writing anything.
# 2. Outputs are pre-filled with a sentinel no identity can produce, and every
#    slot is checked to have been overwritten. A missing write is the bug class
#    here, and it is invisible to `diff` (no crash) and can slip past a plain
#    reference compare — a freshly-zeroed output reads back `0`, which IS the
#    correct answer for `sum`. The sentinel makes "never written" distinct from
#    "correct" for every op, including the case a poisoned allocation cannot
#    reach: empty-axis float `mean` is legitimately NaN, so an unwritten NaN
#    and a correct NaN are indistinguishable without it.
#
# `reduce_product` is deliberately absent: an fp64 reference for a long product
# of uniform values disagrees with the fp32 kernel once intermediates underflow
# to zero, which is a tolerance argument rather than a bug, and the identity it
# would exercise (`1`) is already covered by the hand-written regressions.

from max.gpu.globals import WARP_SIZE
from std.math import isfinite
from std.random import random_ui64, seed
from std.sys import align_of
from std.sys.defines import get_defined_int

from algorithm.reductions import (
    reduce_argmax,
    reduce_argmin,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_sum,
)
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.utils.index import Index, IndexList
from std.utils.numerics import inf, max_finite, min_finite

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

comptime rd_type = DType.float32
comptime idx_type = DType.int64

# The reduced axis's interesting modulus. The rowwise GPU tier ladder pivots on
# `WARP_SIZE * simd_width * chunk_cap`, and the known correctness cliffs are at
# SIMD divisibility of the row (see the warp-tier divisibility-cliff KB entry).
# `WARP_SIZE` resolves per accelerator (32 on NVIDIA and RDNA, 64 on CDNA), so
# the boundary classes land on the pivot of the device under test rather than
# on NVIDIA's pivot everywhere.
comptime OTHER_TILE = 8

# Upper bounds are compile-time so a memory-safety sweep can push the axis past
# the split-K floor (`_SPLITK_MIN_ROW`, 32768) without slowing the fp64
# reference that the `ref` oracle runs on every case. See the skill's
# compile-time-define note: build with `--mojocopt=-D --mojocopt=axis_hi=40960`
# and run `fuzz.py --no-build`.
comptime axis_hi = get_defined_int["axis_hi", 4096]()
comptime other_hi = get_defined_int["other_hi", 16]()

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

comptime OP_SUM = 0
comptime OP_MAX = 1
comptime OP_MIN = 2
comptime OP_MEAN = 3
comptime OP_ARGMAX = 4
comptime OP_ARGMIN = 5
comptime NUM_OPS = 6

# Neither value is reachable as a result: no monoid identity is 111, and no
# valid index is negative. A slot still holding one after the launch was never
# written.
comptime OUT_SENTINEL = Scalar[rd_type](111)
comptime IDX_SENTINEL = Scalar[idx_type](-12345)

# `_reduce_ref` marks "no candidate ever won" (an empty axis, an all-NaN row, a
# row of pure identity values) with this, for which the arg reductions'
# documented contract is index 0.
comptime NO_WINNER = Scalar[idx_type](-1)


def op_name(op: Int) -> String:
    if op == OP_MAX:
        return "max"
    if op == OP_MIN:
        return "min"
    if op == OP_MEAN:
        return "mean"
    if op == OP_ARGMAX:
        return "argmax"
    if op == OP_ARGMIN:
        return "argmin"
    return "sum"


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rank: Int
    var reduce_dim: Int
    var d0: Int
    var d1: Int
    var d2: Int
    var op: Int
    var dist: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "rank=",
            self.rank,
            " reduce_dim=",
            self.reduce_dim,
            " d0=",
            self.d0,
            " d1=",
            self.d1,
            " d2=",
            self.d2,
            " op=",
            op_name(self.op),
            " dist=",
            value_dist_name(self.dist),
        )


def gen_specs(n: Int) raises -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        # (rank, reduce_dim) is compile-time in `reduce_*`, so only the four
        # structurally distinct shapes below are instantiated: rank 1 (the
        # output count is the empty product), rank 2 non-inner and inner (the
        # two dispatch families), and rank 3 non-inner (a non-inner axis with a
        # contiguous inner dim, which is what reaches the tiled tier).
        var combo = Int(random_ui64(0, 3))
        var rank: Int
        var dim: Int
        if combo == 0:
            rank = 1
            dim = 0
        elif combo == 1:
            rank = 2
            dim = 0
        elif combo == 2:
            rank = 2
            dim = 1
        else:
            rank = 3
            dim = 1

        # `d0` is assigned on every branch below, so it takes no initializer
        # (CI builds with -Werror and a dead store is an error there). `d1` and
        # `d2` keep theirs: the lower-rank shapes leave them at 1.
        var d0: Int
        var d1 = 1
        var d2 = 1
        if rank == 1:
            d0 = boundary_int(0, axis_hi, WARP_SIZE)
        elif rank == 2:
            if dim == 0:
                d0 = boundary_int(0, axis_hi, WARP_SIZE)
                d1 = boundary_int(1, other_hi, OTHER_TILE)
            else:
                d0 = boundary_int(1, other_hi, OTHER_TILE)
                d1 = boundary_int(0, axis_hi, WARP_SIZE)
        else:
            d0 = boundary_int(1, other_hi, OTHER_TILE)
            d1 = boundary_int(0, axis_hi, WARP_SIZE)
            d2 = boundary_int(1, other_hi, OTHER_TILE)

        var op = Int(random_ui64(0, NUM_OPS - 1))
        # Same bias as `fuzz_softmax`: mostly uniform, the rest spread over
        # normal/sparse/large/all-equal. NaN/Inf "specials" (id 5) stay out of
        # the auto-mix — NaN-versus-reference is a separate contract — but
        # remain reachable with `--dist 5`.
        var dist = 0 if Int(random_ui64(0, 2)) != 0 else Int(random_ui64(0, 4))
        specs.append(CaseSpec(rank, dim, d0, d1, d2, op, dist))
    return specs^


def _reduce_ref(
    src: Span[Scalar[rd_type], _],
    ref_vals: Span[mut=True, Scalar[rd_type], _],
    ref_idx: Span[mut=True, Scalar[idx_type], _],
    outer: Int,
    axis_size: Int,
    inner: Int,
    op: Int,
):
    """FP64 CPU reference mirroring the monoid semantics exactly.

    `ref_idx[row]` is `NO_WINNER` when no candidate ever won the tie-break —
    an empty axis, an all-NaN row, or a row of pure identity values — because
    `ArgMax`/`ArgMin` compare with a strict `>` / `<` that NaN and the identity
    both lose. `ref_vals[row]` then holds the monoid identity, which is what
    the kernel must emit.
    """
    var sum_id = Float64(0)
    var max_id = Float64(min_finite[rd_type]())
    var min_id = Float64(max_finite[rd_type]())

    for o in range(outer):
        for k in range(inner):
            var row = o * inner + k
            var acc = sum_id
            var best_max = max_id
            var best_min = min_id
            var idx_max = NO_WINNER
            var idx_min = NO_WINNER

            for a in range(axis_size):
                var v = src[(o * axis_size + a) * inner + k].cast[
                    DType.float64
                ]()
                acc += v
                # Strict compares, so NaN never wins and the FIRST occurrence
                # of a repeated extremum keeps the slot — the same rule the
                # monoid's min-index tie-break lands on.
                if v > best_max:
                    best_max = v
                    idx_max = Scalar[idx_type](a)
                if v < best_min:
                    best_min = v
                    idx_min = Scalar[idx_type](a)

            if op == OP_SUM:
                ref_vals[row] = acc.cast[rd_type]()
            elif op == OP_MEAN:
                # Integer dtypes substitute a divisor; float lets IEEE-754
                # answer, and `0 * inf` is the NaN numpy.mean reports.
                if axis_size == 0:
                    ref_vals[row] = (Float64(0) * inf[.float64]()).cast[
                        rd_type
                    ]()
                else:
                    ref_vals[row] = (acc / Float64(axis_size)).cast[rd_type]()
            elif op == OP_MAX:
                ref_vals[row] = best_max.cast[rd_type]()
            elif op == OP_MIN:
                ref_vals[row] = best_min.cast[rd_type]()
            elif op == OP_ARGMAX:
                ref_vals[row] = best_max.cast[rd_type]()
                ref_idx[row] = idx_max
            else:
                ref_vals[row] = best_min.cast[rd_type]()
                ref_idx[row] = idx_min


def _check_arg_outputs(
    src: Span[Scalar[rd_type], _],
    got: Span[Scalar[idx_type], _],
    ref_vals: Span[Scalar[rd_type], _],
    ref_idx: Span[Scalar[idx_type], _],
    outer: Int,
    axis_size: Int,
    inner: Int,
) raises:
    """Tie-tolerant arg-reduction check.

    Comparing the selected index directly would false-positive on any repeated
    extremum, so this verifies the CONTRACT instead: the index is in range, and
    the value it points at is the reference extremum. When no candidate won,
    the contract is exactly index 0 (`_in_range_index`'s clamp).
    """
    for o in range(outer):
        for k in range(inner):
            var row = o * inner + k
            var g = got[row]
            if ref_idx[row] == NO_WINNER:
                if g != 0:
                    raise Error(
                        (
                            "arg reduction with no winning candidate must"
                            " report index 0, got "
                        ),
                        g,
                        " at row ",
                        row,
                        " (axis_size=",
                        axis_size,
                        ")",
                    )
                continue
            if g < 0 or Int(g) >= axis_size:
                raise Error(
                    "arg reduction index out of range: ",
                    g,
                    " at row ",
                    row,
                    " (axis_size=",
                    axis_size,
                    ")",
                )
            var picked = src[(o * axis_size + Int(g)) * inner + k]
            if picked != ref_vals[row]:
                raise Error(
                    "arg reduction selected a non-extremal element: index ",
                    g,
                    " holds ",
                    picked,
                    " but the extremum is ",
                    ref_vals[row],
                    " at row ",
                    row,
                )


def _run_shaped[
    nd: Int, dim: Int, dtype: DType = rd_type
](
    ctx: DeviceContext,
    shape: IndexList[nd],
    op: Int,
    dist: Int,
    check: Bool,
) raises:
    var axis_size = shape[dim]
    var total = shape.flattened_length()

    var num_outputs = 1
    comptime for i in range(nd):
        if i != dim:
            num_outputs *= shape[i]

    var inner = 1
    comptime for i in range(dim + 1, nd):
        inner *= shape[i]
    var outer = 1
    comptime for i in range(dim):
        outer *= shape[i]

    # An empty axis makes `total` zero; a zero-byte device allocation is not
    # meaningful, and nothing reads the slot.
    var in_h = ctx.enqueue_create_host_buffer[rd_type](max(total, 1))
    var out_h = ctx.enqueue_create_host_buffer[rd_type](num_outputs)
    var out_idx_h = ctx.enqueue_create_host_buffer[idx_type](num_outputs)
    ctx.synchronize()
    fill_by_dist(in_h.as_span(), dist)
    for i in range(num_outputs):
        out_h[i] = OUT_SENTINEL
        out_idx_h[i] = IDX_SENTINEL

    var in_d = ctx.enqueue_create_buffer[rd_type](max(total, 1))
    var out_d = ctx.enqueue_create_buffer[rd_type](num_outputs)
    var out_idx_d = ctx.enqueue_create_buffer[idx_type](num_outputs)
    ctx.enqueue_copy(in_d, in_h)
    ctx.enqueue_copy(out_d, out_h)
    ctx.enqueue_copy(out_idx_d, out_idx_h)

    var in_buf = TileTensor(in_d, row_major(Coord(shape)))
    var out_ptr = out_d.unsafe_ptr()
    var out_idx_ptr = out_idx_d.unsafe_ptr()

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
        var idx = in_buf.layout(coords)
        return rebind[SIMD[dtype, width]](
            in_buf.raw_load[
                width=width, alignment=alignment * align_of[rd_type]()
            ](idx)
        )

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[dtype, width]) {var out_ptr, var shape}:
        # `Row.emit` zeroes the reduced axis rather than dropping it, so the
        # output row is the row-major flatten over the OTHER dims.
        var row = 0
        comptime for i in range(nd):
            if i != dim:
                row = row * shape[i] + Int(coords[i].value())
        out_ptr.unsafe_store[width=width](
            row, rebind[SIMD[rd_type, width]](val)
        )

    @always_inline
    def idx_output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[idx_type, width]) {var out_idx_ptr, var shape}:
        var row = 0
        comptime for i in range(nd):
            if i != dim:
                row = row * shape[i] + Int(coords[i].value())
        out_idx_ptr.unsafe_store[width=width](row, val)

    var coord = Coord(shape)
    if op == OP_SUM:
        reduce_sum[dtype, target="gpu", reduce_dim=dim](
            input_fn, output_fn, coord, ctx
        )
    elif op == OP_MAX:
        reduce_max[dtype, target="gpu", reduce_dim=dim](
            input_fn, output_fn, coord, ctx
        )
    elif op == OP_MIN:
        reduce_min[dtype, target="gpu", reduce_dim=dim](
            input_fn, output_fn, coord, ctx
        )
    elif op == OP_MEAN:
        reduce_mean[dtype, target="gpu", reduce_dim=dim](
            input_fn, output_fn, coord, ctx
        )
    elif op == OP_ARGMAX:
        reduce_argmax[dtype, target="gpu", reduce_dim=dim](
            input_fn, idx_output_fn, coord, ctx
        )
    else:
        reduce_argmin[dtype, target="gpu", reduce_dim=dim](
            input_fn, idx_output_fn, coord, ctx
        )
    ctx.synchronize()

    if check:
        var got_h = ctx.enqueue_create_host_buffer[rd_type](num_outputs)
        var got_idx_h = ctx.enqueue_create_host_buffer[idx_type](num_outputs)
        ctx.enqueue_copy(got_h, out_d)
        ctx.enqueue_copy(got_idx_h, out_idx_d)
        ctx.synchronize()

        var ref_vals = ctx.enqueue_create_host_buffer[rd_type](num_outputs)
        var ref_idx = ctx.enqueue_create_host_buffer[idx_type](num_outputs)
        ctx.synchronize()
        for i in range(num_outputs):
            ref_idx[i] = NO_WINNER
        _reduce_ref(
            in_h.as_span(),
            ref_vals.as_span(),
            ref_idx.as_span(),
            outer,
            axis_size,
            inner,
            op,
        )

        var is_arg = op == OP_ARGMAX or op == OP_ARGMIN

        # Written contract, first and independent of any tolerance: a slot
        # still holding the sentinel was never written. This is the whole
        # reason the target exists, and it is the only check that catches the
        # empty-axis `mean` case, whose correct answer (NaN) a value compare
        # exempts from the finite contract.
        for i in range(num_outputs):
            if is_arg:
                if got_idx_h[i] == IDX_SENTINEL:
                    raise Error(
                        "output slot ",
                        i,
                        " of ",
                        num_outputs,
                        " was never written (axis_size=",
                        axis_size,
                        ", op=",
                        op_name(op),
                        ")",
                    )
            elif got_h[i] == OUT_SENTINEL and ref_vals[i] != OUT_SENTINEL:
                raise Error(
                    "output slot ",
                    i,
                    " of ",
                    num_outputs,
                    " was never written (axis_size=",
                    axis_size,
                    ", op=",
                    op_name(op),
                    ")",
                )

        if is_arg:
            _check_arg_outputs(
                in_h.as_span(),
                got_idx_h.as_span(),
                ref_vals.as_span(),
                ref_idx.as_span(),
                outer,
                axis_size,
                inner,
            )
        else:
            # Tolerance from the math, not a global constant. `max`/`min` are
            # exact selections. `sum`/`mean` accumulate in fp32 across a tier's
            # own order, so the honest bound is the forward error
            # `n * eps * max|x|` with a factor for reassociation; on a
            # cancelling `large` distribution that bound is deliberately loose,
            # which costs nothing here because the written-contract check above
            # is what catches this target's bug class.
            var atol = Float64(1e-3)
            var rtol = Float64(1e-2)
            var allow_nonfinite = False
            if op == OP_MAX or op == OP_MIN:
                atol = 0.0
                rtol = 0.0
            elif op == OP_SUM or op == OP_MEAN:
                var max_abs = Float64(0)
                for i in range(total):
                    var v = in_h[i].cast[.float64]()
                    var a = -v if v < 0 else v
                    if isfinite(a) and a > max_abs:
                        max_abs = a
                var bound = 8.0 * 1.1920929e-7 * Float64(axis_size) * max_abs
                if op == OP_MEAN and axis_size > 0:
                    bound /= Float64(axis_size)
                atol = bound
                rtol = 1e-5
                # `_reduce_ref` sums in f64 and only narrows at the end, so it
                # divides (mean) or cancels (sum) its way back into range. The
                # kernel has no such luxury: once a running fp32 partial passes
                # `max_finite` it saturates, and +-Inf is absorbing. Whenever
                # the worst-case partial `n * max|x|` clears that ceiling the
                # overflow is a property of the format rather than of the
                # kernel, so the finite contract has nothing to say about it.
                var ceiling = Float64(max_finite[rd_type]())
                allow_nonfinite = Float64(axis_size) * max_abs > ceiling
            if not numeric_check(
                got_h.as_span(),
                ref_vals.as_span(),
                atol=atol,
                rtol=rtol,
                allow_nonfinite=allow_nonfinite,
            ):
                raise Error(op_name(op), " numeric mismatch")

    _ = in_d
    _ = out_d
    _ = out_idx_d
    _ = in_buf


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    # Clamp the structural fields into the instantiated domain rather than
    # raising on a spec outside it. The shrinker walks every Int field toward 0
    # without knowing which combinations exist, so a raise here would read as
    # the bug still reproducing and the "minimal" repro would be a spec this
    # target cannot run at all. Clamping keeps the shrink honest: an all-zero
    # spec lands on rank 1 / axis 0 / sum, which is the genuine minimal repro.
    var rank = max(1, min(3, spec.rank))
    var dim = max(0, min(rank - 1, spec.reduce_dim))
    if rank == 3:
        dim = 1  # only the non-inner rank-3 form is instantiated

    # Only the reduced axis may be empty; a zero elsewhere means no outputs at
    # all, which is a different (and trivially correct) case.
    var e0 = max(0, spec.d0)
    var e1 = max(0, spec.d1)
    var e2 = max(0, spec.d2)
    var op = max(0, min(NUM_OPS - 1, spec.op))
    var dist = max(0, spec.dist)

    if rank == 1:
        _run_shaped[1, 0](ctx, Index(e0), op, dist, check)
    elif rank == 2 and dim == 0:
        _run_shaped[2, 0](ctx, Index(e0, max(1, e1)), op, dist, check)
    elif rank == 2:
        _run_shaped[2, 1](ctx, Index(max(1, e0), e1), op, dist, check)
    else:
        _run_shaped[3, 1](
            ctx, Index(max(1, e0), e1, max(1, e2)), op, dist, check
        )


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "rank=",
                specs[i].rank,
                "reduce_dim=",
                specs[i].reduce_dim,
                "d0=",
                specs[i].d0,
                "d1=",
                specs[i].d1,
                "d2=",
                specs[i].d2,
                "op=",
                specs[i].op,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var spec = CaseSpec(
            flag_int(args, "--rank", 2),
            flag_int(args, "--reduce_dim", 1),
            flag_int(args, "--d0", 8),
            flag_int(args, "--d1", 128),
            flag_int(args, "--d2", 1),
            flag_int(args, "--op", OP_SUM),
            flag_int(args, "--dist", 0),
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_reductions seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
