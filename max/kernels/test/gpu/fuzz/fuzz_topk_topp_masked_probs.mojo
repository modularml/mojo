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
# Fuzz target: `nn.sampling.topk_topp_masked_probs`, the target-side kernel of
# speculative decoding. It writes each row's top-k/top-p masked renormalized
# softmax, recovered through the dual-pivot cutoff search. The dispatcher
# routes NVIDIA SM90+ builds to a thread-block-cluster kernel and everything
# else (including rows whose per-CTA slice exceeds the shared-memory budget)
# to the single-block kernel, so both launch shapes are reachable from the
# same spec axes:
#
#   - `d` sweeps the vectorization boundary (vec_size = gcd(8, d)), the
#     per-CTA chunk boundary (block_size * vec = 8192), and occasionally the
#     shared-memory gate (~221k elements at cluster width 4), past which the
#     single-block fallback runs.
#   - `rows` sweeps single-wave and multi-wave cluster grids (4 CTAs per row).
#
# Oracle (`--check`, the `ref` default): a TIE- AND BOUNDARY-TOLERANT contract
# validated against an f64 host recompute, per the per-token accept predicate
# the kernel documents (token survives iff `count(e > e_t) < k` and
# `mass(e > e_t) <= p_eff`):
#
#   1. Order: no zeroed token's weight exceeds a kept token's weight (up to
#      f32-vs-f64 rounding of `exp`).
#   2. The boundary KEPT token satisfies the accept predicate with slack; the
#      predicate is monotone in the weight, so that covers every kept token.
#   3. The boundary ZEROED token violates the predicate with slack (the cutoff
#      is minimal). Skipped for weights that underflow f32 (the kernel
#      legitimately zeroes them at p = 1 where f64 would keep them).
#   4. Values: the row equals `e_i / kept_mass` over the KERNEL'S OWN kept
#      set (via `numeric_check`), so a membership flip within rounding slack
#      of the boundary cannot false-positive the value check, while a
#      wrong-by-a-margin membership still fails checks 2/3.
#
# Exact-sequence comparison against a host selection is deliberately avoided:
# near the nucleus boundary, f32 kernel weights and f64 host weights order
# ties differently, and both selections are valid (see the tie-tolerance rule
# in the fuzzing guide).

from std.math import exp
from std.random import random_ui64, seed as set_seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from std.utils.numerics import isfinite
from layout import TileTensor, row_major
from nn.sampling import topk_topp_masked_probs

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

# The rejection sampler's target-side mask runs at float32 under
# `draft_proposal: argmax` and bfloat16 under `sampled`. The dtype
# specializes the kernel, so it is a build-time axis.
comptime in_bf16 = get_defined_int["ttmp_bf16", 0]() == 1
comptime in_type = DType.bfloat16 if in_bf16 else DType.float32
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()

# Relative slack for boundary decisions: the kernel builds its weights as
# f32 `exp((x - m) / T)`; against the f64 recompute the argument rounding
# costs up to ~|arg| * 2^-24 relative, and |arg| reaches ~88 before underflow.
comptime VALUE_SLACK = 1e-4
# Relative slack on the top-p mass comparison: the kernel's masses are f32
# sums of up to ~262k terms.
comptime MASS_SLACK = 1e-3
# Weights below this fraction of the row mass underflow the kernel's f32
# working domain; their membership is not comparable against f64.
comptime UNDERFLOW_FRACTION = 1e-30


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows: Int
    var d: Int  # vocabulary width (vec boundary 8; >221184 -> single-block)
    var k: Int  # top-k; 0 disables (the decode top-p-only shape)
    var top_p_milli: Int  # top-p * 1000 (1000 = no nucleus constraint)
    var temp_milli: Int  # temperature * 1000 (0 = greedy, clamped to 1e-6)
    var dist: Int  # value-distribution id (finite dists only; see gen_specs)

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
            " dist=",
            value_dist_name(self.dist),
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    """Specs across the dispatch space.

    d rolls between the fine vectorization boundary, the per-CTA chunk
    boundary, and (rarely) past the shared-memory gate where the single-block
    kernel serves the row. Value distributions stay finite: the kernel's
    contract does not define output for NaN/Inf logits (the sampler variant
    owns that path).
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var rows = boundary_int(1, 192, 8)
        var d_roll = Int(random_ui64(0, 9))
        var d: Int
        if d_roll < 6:
            d = boundary_int(1, 8192, 8)
        elif d_roll < 9:
            d = boundary_int(8192, 65536, 8192)
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
        var dist = Int(random_ui64(0, 4))  # uniform..all_equal, no specials
        specs.append(CaseSpec(rows, d, k, top_p_milli, temp_milli, dist))
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
    k: Int,
    top_p: Float64,
    temperature: Float64,
    row: Int,
) raises:
    """Validates one output row against the f64 accept-predicate contract."""
    var d = len(logits)
    var m = logits[0]
    for i in range(1, d):
        m = max(m, logits[i])

    var inv_temp = 1.0 / max(temperature, 1e-6)
    var e = List[Float64]()
    var z = Float64(0)
    for i in range(d):
        var v = exp((logits[i] - m) * inv_temp)
        e.append(v)
        z += v
    var k_eff = d if (k <= 0 or k > d) else k
    var p_eff = top_p.clamp(0.0, 1.0) * z

    # Sorted descending weights with prefix masses, for O(log d) predicate
    # evaluation at any threshold.
    var sorted_e = e.copy()

    def _greater(lhs: Float64, rhs: Float64) -> Bool:
        return lhs > rhs

    sort(sorted_e, _greater)
    var prefix = List[Float64]()
    var acc = Float64(0)
    for i in range(d):
        acc += sorted_e[i]
        prefix.append(acc)

    # The kernel's own selection: kept = nonzero output.
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

    # 1. Order: a zeroed weight above a kept weight means the mask is not a
    # threshold at all.
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

    # 2. The boundary kept token must satisfy the accept predicate.
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

    # 3. The boundary zeroed token must violate the predicate (the cutoff is
    # minimal). Not comparable when the weight underflows the kernel's f32
    # domain.
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

    # 4. Values against the kernel's own kept set.
    var expected = List[Float32]()
    for i in range(d):
        if dist_row[i] != 0:
            expected.append(Float32(e[i] / kept_mass))
        else:
            expected.append(Float32(0))
    if not numeric_check(Span(dist_row), Span(expected), atol=1e-6, rtol=1e-3):
        raise Error("row " + String(row) + ": masked distribution mismatch")


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
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    var temp_host = ctx.enqueue_create_host_buffer[.float32](rows)
    for r in range(rows):
        top_p_host[r] = top_p
        top_k_host[r] = Int64(spec.k)
        temp_host[r] = temp
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](rows)
    var temp_dev = ctx.enqueue_create_buffer[.float32](rows)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(temp_dev, temp_host)

    # FRESH output every case, so an unwritten element is visible to the
    # poison / initcheck / memcheck oracles.
    var probs_dev = ctx.enqueue_create_buffer[.float32](in_len)

    topk_topp_masked_probs(
        ctx,
        TileTensor(in_dev, row_major(rows, d)),
        TileTensor(probs_dev, row_major(rows, d)).as_unsafe_any_origin(),
        top_k_val=spec.k,
        top_p_val=top_p,
        top_k_arr=TileTensor(top_k_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        top_p_arr=TileTensor(top_p_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        temperature=TileTensor(temp_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
    )

    var probs_host = ctx.enqueue_create_host_buffer[.float32](in_len)
    ctx.enqueue_copy(probs_host, probs_dev)
    ctx.synchronize()

    if check:
        for r in range(rows):
            var logits = List[Float64]()
            var dist_row = List[Float32]()
            for c in range(d):
                logits.append(Float64(in_host[r * d + c]))
                dist_row.append(probs_host[r * d + c])
            _check_row(
                logits,
                dist_row,
                spec.k,
                Float64(spec.top_p_milli) / 1000.0,
                Float64(spec.temp_milli) / 1000.0,
                r,
            )
    else:
        # Even without the reference, a non-finite output is a contract
        # violation on finite logits (all generated dists are finite).
        for i in range(in_len):
            if not isfinite(probs_host[i]):
                print("FUZZ_CONTRACT_FAIL idx=", i, "value=", probs_host[i])
                raise Error("non-finite masked probability on finite logits")

    _ = in_dev
    _ = top_p_dev
    _ = top_k_dev
    _ = temp_dev
    _ = probs_dev


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
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var rows = flag_int(args, "--rows", 8)
        var d = flag_int(args, "--d", 1024)
        var k = flag_int(args, "--k", 0)
        var top_p_milli = flag_int(args, "--top_p_milli", 950)
        var temp_milli = flag_int(args, "--temp_milli", 1000)
        var dist = flag_int(args, "--dist", 1)
        var spec = CaseSpec(rows, d, k, top_p_milli, temp_milli, dist)
        print("FUZZ_SINGLE", spec)
        with DeviceContext() as ctx:
            run_one_case(ctx, spec, check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_topk_topp_masked_probs seed=",
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
