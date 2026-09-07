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
# Fuzz target: `nn.sampling.topk_topp_sampling_from_prob` with `from_logits`
# and `emit_dist` -- the draft side of speculative decoding. The kernel draws
# one token per row under the joint top-k/top-p constraint and also writes the
# masked renormalized distribution it drew from; spec decode subtracts that
# distribution for its rejection residual and reads the sampled token's
# probability back out of it. The dispatcher routes NVIDIA SM90+ builds to a
# thread-block-cluster kernel (staged shared-memory slices, cluster-parallel
# rejection loop) and everything else -- including rows past the
# shared-memory gate (~221k elements) -- to the single-block fused kernel.
#
# Contracts, in increasing strength:
#
#   ALWAYS (any oracle, incl. `diff`): the sampled token id is in [0, d).
#
#   `--check` (the `ref` default):
#   1. The sampled token carries POSITIVE mass in the emitted distribution --
#      the invariant spec decode relies on. A sampled token the emitted
#      nucleus excludes gives it probability zero and corrupts the rejection
#      residual (the exact hazard the fused kernel design exists to prevent).
#   2. The emitted row is a valid top-k/top-p masked softmax: the same
#      tie- and boundary-tolerant accept-predicate contract as the
#      masked-probs fuzz target, extended with the sampler's inline min-p
#      mask (weights below `min_p` leave the working distribution; the total
#      mass `z` stays unmasked).
#   3. A degenerate row -- one holding a NaN or +Inf logit (some weights go
#      NaN, the loop cannot accept) or made of -Inf only (every weight goes
#      zero, the loop accepts over zero mass) -- must emit an ALL-ZERO
#      distribution row. Its token is bound only by the ALWAYS contract:
#      the loop's fallback token for such rows is arbitrary.
#   4. Cross-path nucleus agreement: a second in-process launch with
#      `emit_dist=False` must sample a token the emitted distribution
#      gives POSITIVE mass. `emit_dist` is a comptime dispatch switch,
#      not a flag on one kernel -- True selects the cluster kernel
#      (`topk_topp_sampling_emit_dist_cluster_*`, what speculative
#      decoding's draft proposal runs), False the single-block one
#      (`topk_topp_sampling_from_prob_*`, what the main token sampler
#      runs). They are separate implementations whose rejection loops
#      walk candidates in different orders, so one seed need not reach
#      the same id in both, and comparing ids only measures that. What
#      must agree is the NUCLEUS: truncation is a property of the
#      logits and the sampling params, not of the decomposition. This
#      is also the only contract that constrains the single-block
#      path's truncation at all, since that path emits no distribution
#      of its own to check. Zero-dist rows are exempt: their token is
#      unconstrained (see contract 3).

from std.math import exp
from std.random import random_ui64, seed as set_seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from std.utils.numerics import isfinite, isnan
from layout import TileTensor, row_major
from nn.sampling import topk_topp_sampling_from_prob

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

# Speculative decoding exercises this kernel at BOTH precisions: float32
# under `draft_proposal: argmax` and bfloat16 under `draft_proposal:
# sampled`, which switches `sampling_logits_dtype` for the whole sampling
# path. The dtype specializes the kernel, so it is a build-time axis.
comptime in_bf16 = get_defined_int["ttsd_bf16", 0]() == 1
comptime in_type = DType.bfloat16 if in_bf16 else DType.float32
comptime out_idx_type = DType.int64
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# See fuzz_topk_topp_masked_probs.mojo for the derivation of the slacks.
comptime VALUE_SLACK = 1e-4
comptime MASS_SLACK = 1e-3
comptime UNDERFLOW_FRACTION = 1e-30


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows: Int
    var d: Int  # vocabulary width (vec boundary 8; >221184 -> single-block)
    var k: Int  # top-k; 0 disables (the decode top-p-only shape)
    var top_p_milli: Int  # top-p * 1000 (1000 = no nucleus constraint)
    var temp_milli: Int  # temperature * 1000 (0 = greedy, clamped to 1e-6)
    var min_p_milli: Int  # min-p * 1000 (0 = no min-p mask)
    var dist: Int  # value-distribution id (incl. specials: NaN/Inf rows)
    var seed: Int  # per-row sampler RNG seeds are seed + row

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "rows=",
            self.rows,
            " d=",
            self.d,
            " k=",
            self.k,
            " top_p_milli=",
            self.top_p_milli,
            " temp_milli=",
            self.temp_milli,
            " min_p_milli=",
            self.min_p_milli,
            " dist=",
            value_dist_name(self.dist),
            " seed=",
            self.seed,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    """Specs across the dispatch space and the degenerate-input contract.

    Distributions include VD_SPECIALS (~1/6 of cases): unlike the masked-probs
    kernel, the sampler defines behavior for NaN/Inf rows (an in-range token
    and a zero distribution), so specials probe a real contract rather than
    noise.
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var rows = boundary_int(1, 192, 8)
        var d_roll = Int(random_ui64(0, 9))
        var d: Int
        if d_roll < 5:
            d = boundary_int(2, 8192, 8)
        elif d_roll < 7:
            d = boundary_int(8192, 65536, 8192)
        elif d_roll < 9:
            # Where production vocabularies actually sit -- MiniMax-M3 is
            # 200064, Llama-3 128256, Qwen 152064. Without this band the
            # generator jumps 65536 -> 221185 and never samples a real one.
            d = boundary_int(65536, 221184, 8192)
        else:
            # Past the cluster shared-memory budget: single-block fallback.
            d = boundary_int(221185, 262144, 8192)
        var k = 0 if Int(random_ui64(0, 1)) == 0 else boundary_int(
            1, min(d, 1024), 32
        )
        var p_roll = Int(random_ui64(0, 3))
        var top_p_milli = 1000 if p_roll == 0 else Int(random_ui64(1, 1000))
        var t_roll = Int(random_ui64(0, 9))
        var temp_milli: Int
        if t_roll < 6:
            temp_milli = 1000
        elif t_roll < 7:
            temp_milli = 0
        else:
            temp_milli = Int(random_ui64(1, 3000))
        var min_p_milli = 0 if Int(random_ui64(0, 3)) != 0 else Int(
            random_ui64(1, 500)
        )
        var dist = 5 if Int(random_ui64(0, 5)) == 0 else Int(random_ui64(0, 4))
        var the_seed = Int(random_ui64(1, 1_000_000))
        specs.append(
            CaseSpec(
                rows, d, k, top_p_milli, temp_milli, min_p_milli, dist, the_seed
            )
        )
    return specs^


def _above(
    sorted_e: List[Float64], prefix: List[Float64], threshold: Float64
) -> Tuple[Int, Float64]:
    """(count, mass) of weights strictly greater than `threshold`.

    `sorted_e` is descending and `prefix[i]` holds the mass of its first
    `i + 1` entries, so both come from one binary search.
    """
    var lo = 0
    var hi = len(sorted_e)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_e[mid] > threshold:
            lo = mid + 1
        else:
            hi = mid
    return Tuple[Int, Float64](lo, prefix[lo - 1] if lo > 0 else 0.0)


def _check_row(
    logits: List[Float64],
    dist_row: List[Float32],
    token: Int,
    k: Int,
    top_p: Float64,
    temperature: Float64,
    min_p: Float64,
    row: Int,
) raises:
    """Validates one emitted row against the f64 accept-predicate contract."""
    var d = len(logits)

    # Degenerate contract: a NaN or +Inf logit drives the whole row's weights
    # NaN (row_max is +Inf/NaN), the rejection loop cannot accept, and the
    # kernel emits token 0 with a zero distribution.
    var degenerate = False
    var m = logits[0]
    for i in range(d):
        m = max(m, logits[i])
        if isnan(logits[i]) or (not isfinite(logits[i]) and logits[i] > 0):
            degenerate = True
    # An all--Inf row zeroes every weight instead of NaN-ing it (the
    # kernel's running max starts finite), but lands on the same contract.
    if not isfinite(m):
        degenerate = True
    if degenerate:
        for i in range(d):
            if dist_row[i] != 0:
                print(
                    "FUZZ_CONTRACT_FAIL row=",
                    row,
                    "degenerate dist idx=",
                    i,
                    "value=",
                    dist_row[i],
                )
                raise Error("degenerate row must emit a zero distribution")
        return

    var inv_temp = 1.0 / max(temperature, 1e-6)
    # The working distribution: min-p masks the weights, the total mass `z`
    # stays unmasked (matching the kernel's separate-softmax semantics).
    var e = List[Float64]()
    var z = Float64(0)
    for i in range(d):
        var v = exp((logits[i] - m) * inv_temp)
        z += v
        if min_p > 0 and v < min_p:
            v = 0.0
        e.append(v)
    var k_eff = d if (k <= 0 or k > d) else k
    var p_eff = top_p.clamp(0.0, 1.0) * z

    # The invariant spec decode reads `q` back through.
    if dist_row[token] <= 0:
        print(
            "FUZZ_CONTRACT_FAIL row=",
            row,
            "token=",
            token,
            "token_prob=",
            dist_row[token],
        )
        raise Error("sampled token has no mass in the emitted distribution")

    var sorted_e = e.copy()

    def _greater(lhs: Float64, rhs: Float64) -> Bool:
        return lhs > rhs

    sort(sorted_e, _greater)
    var prefix = List[Float64]()
    var acc = Float64(0)
    for i in range(d):
        acc += sorted_e[i]
        prefix.append(acc)

    var min_kept = Float64.MAX_FINITE
    var max_zeroed = Float64(-1)
    var kept_mass = Float64(0)
    var n_kept = 0
    for i in range(d):
        if dist_row[i] != 0:
            n_kept += 1
            kept_mass += e[i]
            min_kept = min(min_kept, e[i])
        else:
            max_zeroed = max(max_zeroed, e[i])

    if n_kept == 0:
        raise Error("row " + String(row) + ": every token masked out")

    if max_zeroed > min_kept * (1.0 + VALUE_SLACK):
        print(
            "FUZZ_CONTRACT_FAIL row=",
            row,
            "max_zeroed=",
            max_zeroed,
            "min_kept=",
            min_kept,
        )
        raise Error("zeroed token outweighs a kept token")

    var kept_above = _above(sorted_e, prefix, min_kept * (1.0 + VALUE_SLACK))
    if kept_above[0] >= k_eff:
        print(
            "FUZZ_CONTRACT_FAIL row=",
            row,
            "count_above_kept=",
            kept_above[0],
            "k_eff=",
            k_eff,
        )
        raise Error("kept token violates the top-k constraint")
    if kept_above[1] > p_eff * (1.0 + MASS_SLACK):
        print(
            "FUZZ_CONTRACT_FAIL row=",
            row,
            "mass_above_kept=",
            kept_above[1],
            "p_eff=",
            p_eff,
        )
        raise Error("kept token violates the top-p constraint")

    if max_zeroed > 0 and max_zeroed > z * UNDERFLOW_FRACTION:
        var zeroed_above = _above(
            sorted_e, prefix, max_zeroed / (1.0 + VALUE_SLACK)
        )
        var count_ok = zeroed_above[0] >= k_eff
        var mass_ok = zeroed_above[1] > p_eff * (1.0 - MASS_SLACK)
        if not count_ok and not mass_ok:
            print(
                "FUZZ_CONTRACT_FAIL row=",
                row,
                "max_zeroed=",
                max_zeroed,
                "count_above=",
                zeroed_above[0],
                "mass_above=",
                zeroed_above[1],
                "k_eff=",
                k_eff,
                "p_eff=",
                p_eff,
            )
            raise Error("zeroed token satisfies the accept predicate")

    var expected = List[Float32]()
    for i in range(d):
        if dist_row[i] != 0:
            expected.append(Float32(e[i] / kept_mass))
        else:
            expected.append(Float32(0))
    if not numeric_check(Span(dist_row), Span(expected), atol=1e-6, rtol=1e-3):
        raise Error("row " + String(row) + ": emitted distribution mismatch")


def run_one_case(ctx: DeviceContext, spec: CaseSpec, check: Bool) raises:
    var rows = spec.rows
    var d = spec.d
    # Below the kernel's domain (the shrinker probes 0): vacuously pass so a
    # real failure cannot "shrink" into an empty launch.
    if rows <= 0 or d <= 0:
        return

    var in_len = rows * d
    var in_host = ctx.enqueue_create_host_buffer[in_type](in_len)
    fill_by_dist(in_host.as_span(), spec.dist)
    var in_dev = ctx.enqueue_create_buffer[in_type](in_len)
    ctx.enqueue_copy(in_dev, in_host)

    var top_p = Float32(spec.top_p_milli) / 1000.0
    var temp = Float32(spec.temp_milli) / 1000.0
    var min_p = Float32(spec.min_p_milli) / 1000.0
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[out_idx_type](rows)
    var temp_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var min_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    for r in range(rows):
        top_p_host[r] = top_p
        top_k_host[r] = Scalar[out_idx_type](spec.k)
        temp_host[r] = temp
        min_p_host[r] = min_p
        seed_host[r] = UInt64(spec.seed + r)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[out_idx_type](rows)
    var temp_dev = ctx.enqueue_create_buffer[.float32](rows)
    var min_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](rows)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(min_p_dev, min_p_host)
    ctx.enqueue_copy(seed_dev, seed_host)

    # FRESH outputs every case: an unwritten token row or distribution element
    # is visible to the poison / initcheck / memcheck oracles.
    var tokens_dev = ctx.enqueue_create_buffer[out_idx_type](rows)
    var dist_dev = ctx.enqueue_create_buffer[.float32](in_len)

    topk_topp_sampling_from_prob[
        from_logits=True, emit_dist=True, dist_dtype=DType.float32
    ](
        ctx,
        TileTensor(in_dev, row_major(rows, d)),
        TileTensor(tokens_dev, row_major(rows)),
        spec.k,
        top_p_val=top_p,
        rng_seed=TileTensor(seed_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        top_k_arr=TileTensor(top_k_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        top_p_arr=TileTensor(top_p_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        temperature=TileTensor(temp_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        min_p=TileTensor(min_p_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        out_dist=TileTensor(
            dist_dev, row_major(rows, d)
        ).as_unsafe_any_origin(),
    )

    var tokens_host = ctx.enqueue_create_host_buffer[out_idx_type](rows)
    var dist_host = ctx.enqueue_create_host_buffer[.float32](in_len)
    ctx.enqueue_copy(tokens_host, tokens_dev)
    ctx.enqueue_copy(dist_host, dist_dev)
    ctx.synchronize()

    # Validity contract, checked under every oracle.
    for r in range(rows):
        var tok = Int(tokens_host[r])
        if tok < 0 or tok >= d:
            print("FUZZ_CONTRACT_FAIL row=", r, "token=", tok, "d=", d)
            raise Error("sampled token id out of range [0, d)")

    if check:
        for r in range(rows):
            var logits = List[Float64]()
            var dist_row = List[Float32]()
            for c in range(d):
                logits.append(Float64(in_host[r * d + c]))
                dist_row.append(dist_host[r * d + c])
            _check_row(
                logits,
                dist_row,
                Int(tokens_host[r]),
                spec.k,
                Float64(spec.top_p_milli) / 1000.0,
                Float64(spec.temp_milli) / 1000.0,
                Float64(spec.min_p_milli) / 1000.0,
                r,
            )

        # Cross-path nucleus agreement: the single-block path (what
        # `emit_dist=False` dispatches to) must draw inside the nucleus the
        # cluster path emitted. See contract 4.
        var tokens2_dev = ctx.enqueue_create_buffer[out_idx_type](rows)
        topk_topp_sampling_from_prob[from_logits=True](
            ctx,
            TileTensor(in_dev, row_major(rows, d)),
            TileTensor(tokens2_dev, row_major(rows)),
            spec.k,
            top_p_val=top_p,
            rng_seed=TileTensor(seed_dev, row_major(rows))
            .as_unsafe_any_origin()
            .as_immut(),
            top_k_arr=TileTensor(top_k_dev, row_major(rows))
            .as_unsafe_any_origin()
            .as_immut(),
            top_p_arr=TileTensor(top_p_dev, row_major(rows))
            .as_unsafe_any_origin()
            .as_immut(),
            temperature=TileTensor(temp_dev, row_major(rows))
            .as_unsafe_any_origin()
            .as_immut(),
            min_p=TileTensor(min_p_dev, row_major(rows))
            .as_unsafe_any_origin()
            .as_immut(),
        )
        var tokens2_host = ctx.enqueue_create_host_buffer[out_idx_type](rows)
        ctx.enqueue_copy(tokens2_host, tokens2_dev)
        ctx.synchronize()
        for r in range(rows):
            # A zero emitted row carries no nucleus to be inside of: the
            # row is degenerate and its token is unconstrained (NaN weights
            # or zero mass; see _check_row).
            if dist_host[r * d + Int(tokens_host[r])] == 0:
                continue
            var tb = Int(tokens2_host[r])
            if dist_host[r * d + tb] <= 0:
                print(
                    "FUZZ_CONTRACT_FAIL row=",
                    r,
                    "single_block_token=",
                    tb,
                    "its_emitted_mass=",
                    Float64(dist_host[r * d + tb]),
                    "cluster_token=",
                    Int(tokens_host[r]),
                )
                raise Error(
                    "emit_dist=False sampled outside the emitted nucleus"
                )
        _ = tokens2_dev

    _ = in_dev
    _ = top_p_dev
    _ = top_k_dev
    _ = temp_dev
    _ = min_p_dev
    _ = seed_dev
    _ = tokens_dev
    _ = dist_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    set_seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "rows=",
                specs[i].rows,
                "d=",
                specs[i].d,
                "k=",
                specs[i].k,
                "top_p_milli=",
                specs[i].top_p_milli,
                "temp_milli=",
                specs[i].temp_milli,
                "min_p_milli=",
                specs[i].min_p_milli,
                "dist=",
                specs[i].dist,
                "seed=",
                specs[i].seed,
            )
        return

    if mode == "single":
        var rows = flag_int(args, "--rows", 8)
        var d = flag_int(args, "--d", 1024)
        var k = flag_int(args, "--k", 0)
        var top_p_milli = flag_int(args, "--top_p_milli", 950)
        var temp_milli = flag_int(args, "--temp_milli", 1000)
        var min_p_milli = flag_int(args, "--min_p_milli", 0)
        var dist = flag_int(args, "--dist", 1)
        var spec_seed = flag_int(args, "--seed", the_seed)
        var spec = CaseSpec(
            rows, d, k, top_p_milli, temp_milli, min_p_milli, dist, spec_seed
        )
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_topk_topp_sampling_dist seed=",
        the_seed,
        "budget=",
        the_budget,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
