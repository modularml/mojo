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
# Fuzz target: MoE router `single_group_router` (the MiniMax-M3 `n_groups=1`
# routing path -- one block per token, one thread per expert, warp-bitonic
# top-k). It selects the top `n_experts_per_tok` experts by `score + bias`,
# then returns the UN-biased score, optionally normalized over the selected
# set, times `routed_scaling_factor`. A wrong index sends a token to the wrong
# expert -- a large, discrete accuracy error, not a rounding error.
#
# Single-GPU and hardware-independent (the 8-GPU expert-parallel dispatch
# happens AFTER routing). Fuzzes num_tokens + the score value-distribution.
# n_routed_experts / n_experts_per_tok / norm_weights are compile-time
# (`-D n_routed_experts=.. -D n_experts_per_tok=.. -D norm_weights=..`,
# defaulting to the M3 config 128 / 4 / True).
#
# Two oracles:
#   ref (--check 1): host top-k reference (select by score+bias; weight =
#     unbiased score, normalized, scaled). Compares the selected index set and
#     the weights. Auto-mix distributions are kept overflow-safe (uniform/
#     normal/sparse/all-equal) since the weight normalization sum is not
#     overflow-safe; `large`/`specials` are excluded from the ref auto-mix.
#   contract (--contract 1): injects NaN/Inf/large specials and checks the
#     SAFETY invariant -- every token's k expert indices are in
#     [0, n_routed_experts) ALWAYS (an out-of-range index would OOB the
#     downstream expert gather), and DISTINCT for tokens whose scores are all
#     finite (a duplicate is double-routing). NaN/Inf scores break the sort's
#     total order and cannot occur in production (the router input is sigmoid
#     output in [0, 1]), so duplicates on a non-finite token are exempt -- a
#     latent robustness gap, not a live bug.

from std.random import random_ui64, seed
from std.sys.defines import get_defined_bool, get_defined_int
from std.utils.numerics import isfinite, neg_inf

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.moe import single_group_router

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    fill_uniform,
    fill_with_specials,
    flag,
    flag_int,
    value_dist_name,
)

comptime scores_type = DType.float32
comptime bias_type = DType.float32

# COMPTIME router geometry (default = MiniMax-M3). n_routed_experts must be a
# multiple of WARP_SIZE; n_experts_per_tok must be a power of two.
comptime N_EXPERTS = get_defined_int["n_routed_experts", 128]()
comptime TOPK = get_defined_int["n_experts_per_tok", 4]()
comptime NORM = get_defined_bool["norm_weights", True]()
comptime ROUTED_SCALING = 2.0  # M3 routed_scaling_factor

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var num_tokens: Int
    var dist: Int  # score value-distribution id (see _fuzz: VD_*)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "num_tokens=",
            self.num_tokens,
            " dist=",
            value_dist_name(self.dist),
            " n_experts=",
            N_EXPERTS,
            " topk=",
            TOPK,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var num_tokens = boundary_int(1, 4096, 128)
        # ref-safe auto-mix: uniform(0)/normal(1)/sparse(2)/all_equal(4). The
        # router weight-normalization sum is not overflow-safe, so `large`(3)
        # and NaN/Inf `specials`(5) are excluded from the ref mix; specials are
        # driven by the contract oracle, reachable via `--dist 5`.
        var roll = Int(random_ui64(0, 3))
        var dist = 4 if roll == 3 else roll  # 0,1,2 -> as-is; 3 -> all_equal(4)
        specs.append(CaseSpec(num_tokens, dist))
    return specs^


def _check_router(
    actual_idx: Span[Int32, _],
    actual_w: Span[Scalar[scores_type], _],
    scores: Span[Scalar[scores_type], _],
    bias: Span[Scalar[bias_type], _],
    num_tokens: Int,
) -> Bool:
    """Tie-tolerant router correctness check.

    Top-k routing is invariant to tie-breaking: when two experts have
    (near-)equal biased scores, selecting either is correct, so demanding an
    exact index sequence false-positives on ties. Instead verify, per token:

    1. selection validity -- every kernel-selected expert is genuinely within
       the top-k by biased score: its biased score is >= the k-th largest minus
       a tie tolerance. A real mis-route selects an expert below that.
    2. weight correctness -- the returned weights match what the KERNEL's own
       selection implies (unbiased score, normalized over the selected set if
       NORM, scaled), so a legitimate tie swap does not trip it.
    """
    for t in range(num_tokens):
        var base = t * N_EXPERTS
        var kbase = t * TOPK
        var biased = List[Float64]()
        for e in range(N_EXPERTS):
            biased.append(
                scores[base + e].cast[.float64]() + bias[e].cast[.float64]()
            )
        # k-th largest biased score (the top-k membership threshold).
        var work = biased.copy()
        var threshold = neg_inf[.float64]()
        for _ in range(TOPK):
            var best = 0
            var best_val = neg_inf[.float64]()
            for e in range(N_EXPERTS):
                if work[e] > best_val:
                    best_val = work[e]
                    best = e
            threshold = best_val
            work[best] = neg_inf[.float64]()
        var tol = 1e-3 + 1e-3 * abs(threshold)

        # (1) Every selected expert is in the true top-k (up to ties).
        for j in range(TOPK):
            var idx = Int(actual_idx[kbase + j])
            if idx < 0 or idx >= N_EXPERTS:
                print(
                    "FUZZ_NUMERIC_FAIL kind=range token=",
                    t,
                    "slot=",
                    j,
                    "idx=",
                    idx,
                )
                return False
            if biased[idx] < threshold - tol:
                print(
                    "FUZZ_NUMERIC_FAIL kind=topk token=",
                    t,
                    "slot=",
                    j,
                    "idx=",
                    idx,
                    "biased=",
                    biased[idx],
                    "kth_largest=",
                    threshold,
                )
                return False

        # (2) Weights match the kernel's OWN selection.
        var wsum = Float64(0)
        for j in range(TOPK):
            wsum += scores[base + Int(actual_idx[kbase + j])].cast[
                DType.float64
            ]()
        if abs(wsum) < 1e-6:
            continue  # degenerate normalization (all-zero selected): skip
        for j in range(TOPK):
            var w = scores[base + Int(actual_idx[kbase + j])].cast[
                DType.float64
            ]()
            comptime if NORM:
                w /= wsum
            w *= ROUTED_SCALING
            var got = actual_w[kbase + j].cast[.float64]()
            if abs(got - w) > 1e-2 + 1e-2 * abs(w):
                print(
                    "FUZZ_NUMERIC_FAIL kind=weight token=",
                    t,
                    "slot=",
                    j,
                    "got=",
                    got,
                    "expected=",
                    w,
                )
                return False
    return True


def _token_all_finite(scores: Span[Scalar[scores_type], _], t: Int) -> Bool:
    """Whether every score for token `t` is finite."""
    var base = t * N_EXPERTS
    for e in range(N_EXPERTS):
        if not isfinite(scores[base + e]):
            return False
    return True


def _check_index_contract(
    actual_idx: Span[Int32, _],
    scores: Span[Scalar[scores_type], _],
    num_tokens: Int,
) -> Bool:
    """Router safety contract:

    - in-range ALWAYS: every written expert index is in [0, N_EXPERTS). An
      out-of-range index is a memory-safety bug regardless of input (it would
      OOB the downstream expert gather/dispatch).
    - distinct for FINITE tokens only: a token whose scores are all finite must
      get DISTINCT experts (a duplicate is double-routing). NaN/Inf scores break
      the sort's total order and cannot occur in production (router input is
      sigmoid output in [0, 1]), so duplicates on a non-finite token are exempt.
    """
    for t in range(num_tokens):
        var finite = _token_all_finite(scores, t)
        for j in range(TOPK):
            var idx = Int(actual_idx[t * TOPK + j])
            if idx < 0 or idx >= N_EXPERTS:
                print(
                    "FUZZ_CONTRACT_FAIL kind=range token=",
                    t,
                    "slot=",
                    j,
                    "idx=",
                    idx,
                )
                return False
            if finite:
                for j2 in range(j + 1, TOPK):
                    if idx == Int(actual_idx[t * TOPK + j2]):
                        print(
                            "FUZZ_CONTRACT_FAIL kind=dup token=",
                            t,
                            "idx=",
                            idx,
                        )
                        return False
    return True


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
) raises:
    var nt = spec.num_tokens

    var scores_host = ctx.enqueue_create_host_buffer[scores_type](
        nt * N_EXPERTS
    )
    var bias_host = ctx.enqueue_create_host_buffer[bias_type](N_EXPERTS)

    # Scores: contract mode injects NaN/Inf/large specials; otherwise the spec's
    # distribution. Bias: small distinct uniform values (keeps biased scores
    # distinct so the ref top-k order is unambiguous).
    if contract:
        fill_with_specials(scores_host.as_span(), density=0.3)
    else:
        fill_by_dist(scores_host.as_span(), spec.dist)
    fill_uniform(bias_host.as_span(), lo=-0.1, hi=0.1)

    var scores_dev = ctx.enqueue_create_buffer[scores_type](nt * N_EXPERTS)
    var bias_dev = ctx.enqueue_create_buffer[bias_type](N_EXPERTS)
    var idx_dev = ctx.enqueue_create_buffer[.int32](nt * TOPK)
    var w_dev = ctx.enqueue_create_buffer[scores_type](nt * TOPK)
    ctx.enqueue_copy(scores_dev, scores_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    var expert_scores = TileTensor(
        scores_dev, row_major(Coord(nt, Idx[N_EXPERTS]))
    )
    var expert_bias = TileTensor(bias_dev, row_major(Idx[N_EXPERTS]))
    var expert_indices = TileTensor(idx_dev, row_major(Coord(nt, Idx[TOPK])))
    var expert_weights = TileTensor(w_dev, row_major(Coord(nt, Idx[TOPK])))

    single_group_router[N_EXPERTS, TOPK, NORM, "gpu"](
        expert_indices,
        expert_weights,
        expert_scores,
        expert_bias,
        Float32(ROUTED_SCALING),
        ctx,
    )
    ctx.synchronize()

    if contract:
        var idx_h = ctx.enqueue_create_host_buffer[.int32](nt * TOPK)
        ctx.enqueue_copy(idx_h, idx_dev)
        ctx.synchronize()
        if not _check_index_contract(
            idx_h.as_span(), scores_host.as_span(), nt
        ):
            raise Error("MoE router index contract violated")
    elif check:
        var idx_h = ctx.enqueue_create_host_buffer[.int32](nt * TOPK)
        var w_h = ctx.enqueue_create_host_buffer[scores_type](nt * TOPK)
        ctx.enqueue_copy(idx_h, idx_dev)
        ctx.enqueue_copy(w_h, w_dev)
        ctx.synchronize()
        if not _check_router(
            idx_h.as_span(),
            w_h.as_span(),
            scores_host.as_span(),
            bias_host.as_span(),
            nt,
        ):
            raise Error("MoE router top-k/weight mismatch")

    _ = scores_dev
    _ = bias_dev
    _ = idx_dev
    _ = w_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var contract = flag_int(args, "--contract", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "num_tokens=",
                specs[i].num_tokens,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var nt = flag_int(args, "--num_tokens", 64)
        var dist = flag_int(args, "--dist", 0)
        print("FUZZ_SINGLE num_tokens=", nt, "dist=", dist)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(nt, dist), check, contract)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_moe_router seed=",
        the_seed,
        "budget=",
        the_budget,
        "n_experts=",
        N_EXPERTS,
        "topk=",
        TOPK,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, contract)
    print("=== done:", len(specs), "cases ===")
