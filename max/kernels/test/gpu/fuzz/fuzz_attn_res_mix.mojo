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
# Fuzz target: attn_res_mix (`attn_res_mix_gpu`, //Kernels/lib/attn_res).
#
# The kernel runs one CTA per token over a comptime candidate axis: per
# candidate it does a fused RMS-norm-and-score block reduction, softmaxes the
# per-candidate scores in shared memory, then writes the softmax-weighted sum
# of the raw candidates. Everything the fuzzer can reach is runtime-shapeable
# -- including the candidate count, which is dispatched to its comptime value
# by the same 1..MAX_CANDIDATES ladder the production op uses
# (builtin_kernels/attn_res.mojo), so a case never needs a recompile.
#
# Axes and why they are the interesting ones:
#   hidden  -- drives BOTH the grid-stride loop (`for h in range(tid, hidden,
#              BLOCK)`, so BLOCK decides the iteration count and whether the
#              last iteration is ragged) and the reduction length. Drawn
#              against two moduli: BLOCK, and WARP (below which
#              `block_reduce_dual_sum` is carrying whole warps that loaded
#              nothing but must still reach every barrier).
#   cands   -- 1 (softmax degenerates) through the compiled maximum.
#   tokens  -- grid width, from the single-token decode shape up past a wave.
#   pdl     -- launch behind a real producer GRID with no host sync, PDL
#              attribute armed, so the in-kernel `griddepcontrol.wait` is what
#              orders the read. The kernel test covers this at one shape; here
#              it rides along every shape.
#   dt      -- element dtype. The graph op is generic over it
#              (builtin_kernels/attn_res.mojo) and a served K3 instantiates
#              bf16, which is also the specialization whose input casts and
#              output downcast are lossy, so it cannot be an fp32-only target.
#   dist    -- value distribution (see the gen_specs note on which are excluded
#              from the auto-mix and why).
#
# Oracles: memory safety (memcheck/initcheck/redzone/poison) and hangs/crashes
# (diff) need nothing extra. `ref` (--check) compares against an fp64 CPU
# recompute. `contract` (--contract) injects a bounded count of NaN/Inf
# specials (NOT the usual per-element rate -- see _fill_contract_inputs for why
# that is vacuous on a long reduction) and checks the convex-combination
# envelope, which needs no tolerance and so holds on any input. The shared
# `scores` array is written under `if tid == 0` and read by every thread after
# a bare `barrier()`, and the block reduction is a fixed-order in-block tree
# with no atomics, so the output must also be bit-stable run to run --
# `determinism` (--rerun) pins that, and racecheck/synccheck apply directly
# (this kernel is not warp-specialized, so they do not carry the named-barrier
# false-positive noise the README warns about).

from std.math import ceildiv, exp, max, min, sqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
from max.gpu import block_dim, block_idx, thread_idx
from std.utils.numerics import inf, isnan, max_finite, min_finite, nan, neg_inf

from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from layout import TileTensor, row_major

from attn_res.mix import attn_res_mix_gpu

from _fuzz import (
    VD_ALL_EQUAL,
    VD_UNIFORM,
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

comptime BLOCK = get_defined_int["block_size", 256]()  # production launch width
comptime WARP = 32
comptime MAX_CANDIDATES = 8  # the production op's compiled dispatch range
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# Element-dtype axis. The harness carries spec fields as ints (they become
# `--<key> <int>` flags and land verbatim in corpus JSON), so the dtype travels
# as an id and is dispatched to its comptime value in `run_one_case`.
comptime DT_F32 = 0
comptime DT_BF16 = 1
comptime NUM_DTYPES = 2

# Tolerance for `ref`. The kernel accumulates in fp32 (a per-thread serial sum
# over hidden/BLOCK terms, then a warp+block tree) against an fp64 reference,
# and the softmax consuming those scores is Lipschitz with constant <= 1/4, so
# a score error reaches the output damped rather than amplified. Running this
# generator with the tolerance set to zero measures the real gap: worst
# absolute error 3.6e-6 over 104 cases across three seeds, at the top of the
# hidden range where the accumulation is longest. ATOL keeps ~25x headroom on
# that while staying 10x tighter than the harness default; RTOL only governs
# large-magnitude elements, since the near-zero outputs that show a big
# relative error are all inside ATOL.
comptime REF_ATOL_F32 = 1e-4
comptime REF_RTOL_F32 = 1e-3

# bf16 is bounded by its own storage quantum, not by the reduction: the
# accumulation still runs in fp32, and then the result is rounded once on the
# way out. Measured the same way (tolerances zeroed): worst absolute error
# 7.8125e-3 over 88 cases across three seeds -- which is exactly 2^-7, i.e. one
# output ULP at the magnitudes the fills produce, so it is the store's rounding
# and nothing else. RTOL carries it (one ULP is 2^-8 relative, so 2e-2 is ~5x
# headroom); ATOL covers the near-zero outputs where a relative bound has no
# purchase.
comptime REF_ATOL_BF16 = 2e-2
comptime REF_RTOL_BF16 = 2e-2


def _dt_name(dt: Int) -> String:
    if dt == DT_BF16:
        return "bfloat16"
    return "float32"


# How many NaN/Inf/denormal/+-0 specials the `contract` oracle injects per case
# -- a bounded COUNT, not a per-element rate. See _fill_contract_inputs.
comptime SPECIALS_PER_CASE = 8


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var tokens: Int
    var hidden: Int
    var cands: Int
    var dist: Int  # value-distribution id (see _fuzz: VD_*)
    var pdl: Int  # 1 = launch behind a producer grid with PDL armed
    var dt: Int  # element dtype id (DT_F32 / DT_BF16)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "tokens=",
            self.tokens,
            " hidden=",
            self.hidden,
            " cands=",
            self.cands,
            " dist=",
            value_dist_name(self.dist),
            " pdl=",
            self.pdl,
            " dt=",
            _dt_name(self.dt),
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        # Two moduli matter and boundary_int takes one, so alternate: BLOCK is
        # the grid-stride pivot, WARP is where the block reduction starts
        # carrying warps that loaded nothing.
        var hidden: Int
        if Int(random_ui64(0, 2)) != 0:
            hidden = boundary_int(1, 8192, BLOCK)
        else:
            hidden = boundary_int(1, 1024, WARP)

        # Bias to uniform; the rest spread over normal/sparse/all-equal.
        # `large` (VD 3) is deliberately out of the auto-mix, and NOT because
        # the kernel misbehaves there: near-max_finite values overflow the
        # fp32 sum of squares, so the kernel deliberately scores 0 (matching
        # what the unfused graph's elementwise normalize yields -- see the
        # `rms_scale == 0` branch in the kernel), while the fp64 reference
        # never overflows and scores something large and finite. The two
        # disagree by construction, so `ref` would flag a correct kernel. The
        # regime is pinned instead by
        # `test_overflow_matches_unfused_semantics`, which checks it against
        # the unfused fp32 semantics rather than an fp64 recompute.
        # `specials` (VD 5) is likewise the contract oracle's input, not
        # ref's. Both stay reachable via an explicit `--dist`.
        var dist = Int(VD_UNIFORM)
        if Int(random_ui64(0, 2)) == 0:
            var pick = Int(random_ui64(0, 3))
            dist = Int(VD_ALL_EQUAL) if pick == 3 else pick

        specs.append(
            CaseSpec(
                boundary_int(1, 128, 8),
                hidden,
                boundary_int(1, MAX_CANDIDATES, 4),
                dist,
                Int(random_ui64(0, 1)),
                Int(random_ui64(0, UInt64(NUM_DTYPES - 1))),
            )
        )
    return specs^


def _producer_kernel[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    n: Int32,
):
    """Stand-in for the real graph's `concat`: a GPU grid that writes the exact
    buffer `attn_res_mix_gpu` reads next, immediately before it on the same
    stream. A DMA `enqueue_copy` predecessor would not exercise the
    grid-to-grid handshake the kernel's PDL wait exists for."""
    var i = Int32(block_idx.x) * Int32(block_dim.x) + Int32(thread_idx.x)
    if i < n:
        dst[Int(i)] = src[Int(i)]


def _fill_contract_inputs[
    dtype: DType
](
    cands: Span[mut=True, Scalar[dtype], _],
    proj: Span[mut=True, Scalar[dtype], _],
    norm: Span[mut=True, Scalar[dtype], _],
    dist: Int,
):
    """Contract-mode input: a finite base fill plus a BOUNDED number of
    NaN/Inf/denormal/+-0 specials injected into the candidates.

    A per-element specials RATE (what `fill_with_specials` does, and the model
    the other targets use) is wrong for a kernel that reduces over a `hidden`
    of up to 8192: at any density every token's reduction swallows a special,
    every score goes NaN, and the envelope check below then skips every element
    -- the oracle would pass vacuously on an arbitrarily broken kernel. At the
    30% the harness default uses, P(a token stays fully finite) is 1e-30 by
    hidden=96 and underflows to zero past ~800. Bounding the COUNT instead
    leaves most tokens fully finite, so their envelopes stay tight and the
    check keeps its teeth, while the handful of corrupted tokens still
    exercises NaN propagation.

    proj/norm stay finite for the same reason: they are shared across every
    token, so a single special in either poisons every score in the case.
    """
    fill_by_dist(cands, dist)
    fill_by_dist(proj, dist)
    fill_by_dist(norm, dist)

    var specials: List[Scalar[dtype]] = [
        nan[dtype](),
        inf[dtype](),
        neg_inf[dtype](),
        Scalar[dtype](0),
        -Scalar[dtype](0),
        max_finite[dtype](),
        min_finite[dtype](),
    ]
    var n = len(cands)
    var count = min(SPECIALS_PER_CASE, n)
    for _ in range(count):
        var at = Int(random_ui64(0, UInt64(n - 1)))
        cands[at] = specials[Int(random_ui64(0, UInt64(len(specials) - 1)))]


def _attn_res_ref[
    dtype: DType
](
    cands_h: Span[Scalar[dtype], _],
    proj_h: Span[Scalar[dtype], _],
    norm_h: Span[Scalar[dtype], _],
    dst: Span[mut=True, Scalar[dtype], _],
    tokens: Int,
    c_count: Int,
    hidden: Int,
    eps: Float64,
):
    """FP64 CPU reference: RMS-normalize each candidate, score it against
    `proj * norm`, softmax over the candidate axis, and emit the
    softmax-weighted sum of the RAW candidates.

    Independent in PRECISION, not in association: the scale lands after the
    dot over raw `v` here too, exactly as the kernel does it (`dot / sqrt(mean
    + eps)` vs the kernel's `rsqrt(mean + eps) * dot` differs in rounding
    only). So this oracle checks the arithmetic -- accumulation, softmax,
    weighted sum -- but not the reassociation identity itself; the reference
    order in the kernel's docstring (`k = v * rsqrt(...)` elementwise, then
    `dot(k, w)`) is never computed here. The two orders diverge only under
    fp32 overflow, which `test_overflow_matches_unfused_semantics` in the
    kernel test pins against the unfused semantics -- the same regime this
    oracle excludes the `large` distribution for (see `gen_specs`).
    """
    var scores = List[Float64]()
    for _ in range(c_count):
        scores.append(Float64(0))

    for t in range(tokens):
        for c in range(c_count):
            var base = (t * c_count + c) * hidden
            var sum_sq = Float64(0)
            var dot = Float64(0)
            for h in range(hidden):
                var v = cands_h[base + h].cast[.float64]()
                var sw = proj_h[h].cast[.float64]() * norm_h[h].cast[.float64]()
                sum_sq += v * v
                dot += v * sw
            scores[c] = dot / sqrt(sum_sq / Float64(hidden) + eps)

        var max_s = scores[0]
        for c in range(1, c_count):
            if scores[c] > max_s:
                max_s = scores[c]
        var denom = Float64(0)
        for c in range(c_count):
            scores[c] = exp(scores[c] - max_s)
            denom += scores[c]

        for h in range(hidden):
            var acc = Float64(0)
            for c in range(c_count):
                var v = cands_h[(t * c_count + c) * hidden + h].cast[
                    DType.float64
                ]()
                acc += (scores[c] / denom) * v
            dst[t * hidden + h] = acc.cast[dtype]()


def _check_mix_contract[
    dtype: DType
](
    out_h: Span[Scalar[dtype], _],
    cands_h: Span[Scalar[dtype], _],
    tokens: Int,
    c_count: Int,
    hidden: Int,
) -> Bool:
    """Convex-combination envelope: the softmax weights are non-negative and
    sum to 1, so every output element must land inside the [min, max] of the
    candidates it mixes at that position.

    This is value-independent -- unlike a tolerance diff it cannot
    false-positive on a NaN/Inf input, which is why it is the oracle for the
    specials distribution. NaN output is legitimate propagation (an Inf
    anywhere in a candidate row saturates that row's sum-of-squares, driving
    its score to NaN and every softmax weight with it), so it is skipped rather
    than flagged; a NaN in the envelope itself makes the bound vacuous.
    """
    for t in range(tokens):
        for h in range(hidden):
            var v = out_h[t * hidden + h]
            if Bool(isnan(v)):
                continue
            var lo = cands_h[(t * c_count) * hidden + h]
            var hi = lo
            for c in range(1, c_count):
                var x = cands_h[(t * c_count + c) * hidden + h]
                lo = min(lo, x)
                hi = max(hi, x)
            if Bool(isnan(lo)) or Bool(isnan(hi)):
                continue
            # A few ulps of slack: the weighted sum rounds in fp32.
            var mag = max(abs(lo), abs(hi))
            var slack = Scalar[dtype](1e-6) + Scalar[dtype](1e-5) * mag
            if Bool(v < lo - slack) or Bool(v > hi + slack):
                print(
                    "FUZZ_CONTRACT_FAIL t=",
                    t,
                    "h=",
                    h,
                    "val=",
                    v,
                    "lo=",
                    lo,
                    "hi=",
                    hi,
                )
                return False
    return True


def _run_case[
    dtype: DType
](
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
    rerun: Int = 0,
) raises:
    comptime ref_atol = (
        REF_ATOL_F32 if dtype == DType.float32 else REF_ATOL_BF16
    )
    comptime ref_rtol = (
        REF_RTOL_F32 if dtype == DType.float32 else REF_RTOL_BF16
    )

    var tokens = spec.tokens
    var hidden = spec.hidden
    var c_count = spec.cands
    if c_count < 1 or c_count > MAX_CANDIDATES:
        raise Error(
            "attn_res_mix fuzz: cands outside the compiled 1..",
            MAX_CANDIDATES,
            " dispatch range",
        )
    var use_pdl = spec.pdl == 1
    var n_cand = tokens * c_count * hidden
    var n_out = tokens * hidden
    var eps = Float32(1e-6)

    var cands_h = ctx.enqueue_create_host_buffer[dtype](n_cand)
    var proj_h = ctx.enqueue_create_host_buffer[dtype](hidden)
    var norm_h = ctx.enqueue_create_host_buffer[dtype](hidden)
    if contract:
        _fill_contract_inputs(
            cands_h.as_span(), proj_h.as_span(), norm_h.as_span(), spec.dist
        )
    else:
        fill_by_dist(cands_h.as_span(), spec.dist)
        fill_by_dist(proj_h.as_span(), spec.dist)
        fill_by_dist(norm_h.as_span(), spec.dist)

    var cands_d = ctx.enqueue_create_buffer[dtype](n_cand)
    var proj_d = ctx.enqueue_create_buffer[dtype](hidden)
    var norm_d = ctx.enqueue_create_buffer[dtype](hidden)
    var out_d = ctx.enqueue_create_buffer[dtype](n_out)
    ctx.enqueue_copy(proj_d, proj_h)
    ctx.enqueue_copy(norm_d, norm_h)

    # In the PDL case the upload lands in staging and a producer GRID moves it
    # into `cands_d` right before the mix kernel with no sync between, so the
    # only thing ordering the read is the kernel's own griddepcontrol wait.
    var staging_d = ctx.enqueue_create_buffer[dtype](n_cand if use_pdl else 1)
    if use_pdl:
        ctx.enqueue_copy(staging_d, cands_h)
    else:
        ctx.enqueue_copy(cands_d, cands_h)
    ctx.synchronize()

    var cands_tt = TileTensor(cands_d, row_major(tokens, c_count, hidden))
    var proj_tt = TileTensor(proj_d, row_major(1, hidden))
    var norm_tt = TileTensor(norm_d, row_major(hidden))
    var out_tt = TileTensor(out_d, row_major(tokens, hidden))
    var attrs = pdl_launch_attributes(PDLLevel.ON if use_pdl else PDLLevel.OFF)

    var first_h = ctx.enqueue_create_host_buffer[dtype](n_out)
    var rep_h = ctx.enqueue_create_host_buffer[dtype](n_out)

    for run in range(max(1, rerun)):
        if use_pdl:
            ctx.enqueue_function[_producer_kernel[dtype]](
                cands_d.unsafe_ptr(),
                staging_d.unsafe_ptr(),
                Int32(n_cand),
                grid_dim=(ceildiv(n_cand, BLOCK),),
                block_dim=(BLOCK,),
            )

        # Same 1..MAX_CANDIDATES ladder as the production op: the candidate
        # count is a comptime kernel parameter, so it is dispatched here rather
        # than baked as a define, which keeps it a runtime fuzz axis.
        comptime for c in range(1, MAX_CANDIDATES + 1):
            if c_count == c:
                ctx.enqueue_function[
                    attn_res_mix_gpu[
                        dtype,
                        out_tt.LayoutType,
                        out_tt.Engine,
                        cands_tt.LayoutType,
                        cands_tt.Engine,
                        proj_tt.LayoutType,
                        proj_tt.Engine,
                        norm_tt.LayoutType,
                        norm_tt.Engine,
                        c,
                        BLOCK,
                    ]
                ](
                    out_tt,
                    cands_tt,
                    proj_tt,
                    norm_tt,
                    eps,
                    Int32(hidden),
                    grid_dim=(tokens,),
                    block_dim=(BLOCK,),
                    attributes=attrs.copy(),
                )
        ctx.synchronize()

        if rerun > 0:
            # Run-to-run determinism: the block reduction is a fixed-order
            # in-block tree and there are no atomics, so a difference is a real
            # race (a dropped barrier around the shared `scores`), not a
            # legitimate reduction-order wobble.
            if run == 0:
                ctx.enqueue_copy(first_h, out_d)
                ctx.synchronize()
            else:
                ctx.enqueue_copy(rep_h, out_d)
                ctx.synchronize()
                if not numeric_check(
                    rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
                ):
                    raise Error("attn_res_mix run-to-run nondeterminism")

    if contract:
        var out_h = ctx.enqueue_create_host_buffer[dtype](n_out)
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        if not _check_mix_contract(
            out_h.as_span(), cands_h.as_span(), tokens, c_count, hidden
        ):
            raise Error("attn_res_mix convex-combination contract violated")
    elif check:
        var out_h = ctx.enqueue_create_host_buffer[dtype](n_out)
        var ref_h = ctx.enqueue_create_host_buffer[dtype](n_out)
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        _attn_res_ref(
            cands_h.as_span(),
            proj_h.as_span(),
            norm_h.as_span(),
            ref_h.as_span(),
            tokens,
            c_count,
            hidden,
            eps.cast[.float64](),
        )
        if not numeric_check(
            out_h.as_span(),
            ref_h.as_span(),
            atol=ref_atol,
            rtol=ref_rtol,
        ):
            raise Error("attn_res_mix numeric mismatch")

    _ = cands_d
    _ = proj_d
    _ = norm_d
    _ = out_d
    _ = staging_d
    _ = cands_tt
    _ = proj_tt
    _ = norm_tt
    _ = out_tt


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
    rerun: Int = 0,
) raises:
    """Dispatches the runtime dtype id to its comptime dtype.

    Same shape as the candidate-count ladder inside `_run_case`, and for the
    same reason: the kernel takes its dtype as a comptime parameter, so a
    runtime axis has to be resolved here rather than baked in as a define --
    that is what lets one binary cover both dtypes and keeps `dt` a spec field
    the corpus can pin.
    """
    if spec.dt == DT_F32:
        _run_case[.float32](ctx, spec, check, contract, rerun)
    elif spec.dt == DT_BF16:
        _run_case[.bfloat16](ctx, spec, check, contract, rerun)
    else:
        raise Error(
            "attn_res_mix fuzz: unknown dtype id ",
            spec.dt,
            " (compiled: 0..",
            NUM_DTYPES - 1,
            ")",
        )


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var contract = flag_int(args, "--contract", 0) == 1
    var rerun = flag_int(args, "--rerun", 0)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "tokens=",
                specs[i].tokens,
                "hidden=",
                specs[i].hidden,
                "cands=",
                specs[i].cands,
                "dist=",
                specs[i].dist,
                "pdl=",
                specs[i].pdl,
                "dt=",
                specs[i].dt,
            )
        return

    if mode == "single":
        var tokens = flag_int(args, "--tokens", 8)
        var hidden = flag_int(args, "--hidden", 7168)
        var cands = flag_int(args, "--cands", 3)
        var dist = flag_int(args, "--dist", 0)
        var pdl = flag_int(args, "--pdl", 0)
        var dt = flag_int(args, "--dt", DT_F32)
        print(
            "FUZZ_SINGLE tokens=",
            tokens,
            "hidden=",
            hidden,
            "cands=",
            cands,
            "dist=",
            dist,
            "pdl=",
            pdl,
            "dt=",
            dt,
        )
        with DeviceContext() as ctx:
            run_one_case(
                ctx,
                CaseSpec(tokens, hidden, cands, dist, pdl, dt),
                check,
                contract,
                rerun,
            )
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_attn_res_mix seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, contract, rerun)
    print("=== done:", len(specs), "cases ===")
