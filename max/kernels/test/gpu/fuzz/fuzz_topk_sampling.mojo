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
# Fuzz target: token sampler `fused_token_sampling_gpu` (see
# gpu-kernels-fuzzing-design.md). Target hardware family: NVIDIA SM100 (B200),
# but the bugs under test are hardware-independent (control-flow / NaN-ordering
# gaps, not arch intrinsics).
#
# Two dispatch routes of `fused_token_sampling_gpu` are fuzzed (the `gumbel`
# spec field selects which):
#
#   gumbel == 0  -> top-k path. max_k = batch-max per-row k. Batch-max k < 10
#       routes to `topk_gpu[sampling=True]` -> `_topk_stage2`; k in [10, 32)
#       routes to the FI rejection-sampling kernel, so both dispatch routes
#       get fuzzed. The S1 bug on the two-stage route is a per-row
#       top_k == 0 (k_batch == 0) leaving the int64 output row UNWRITTEN
#       (no sentinel fill in the sampling branch).
#
#   gumbel == 1  -> SERVED greedy path. max_k = -1 and min_top_p = 1.0 so the
#       call routes to `gumbel_sampling_gpu` (topk.mojo:2262), which applies
#       Gumbel noise then argmax. Two suspected triggers (this is the
#       production SERVOPT-1458 path -- top_k=-1, top_p=1, per-request temp=0):
#         (a) Unclamped temperature divide: apply_gumbel_noise_kernel does
#             `input_val / temp_val` with NO clamp (topk.mojo:2397), unlike
#             _topk_stage2 which clamps max(temp_val,1e-6) (:1517). temp==0 ->
#             finite/0 -> +-Inf, 0/0 -> NaN. The argmax is a
#             TopK_2.insert `if elem > self.u` (topk.mojo:636-640) seeded with
#             p=-1, u=-Inf. `NaN > x` and `x > NaN` are both False, and
#             `-Inf > -Inf` is False, so a row that produces NO value strictly
#             greater than -Inf (all NaN, or all -Inf because every logit was
#             <= 0 before the temp=0 divide) leaves p == -1 -> the argmax emits
#             token id -1 -> the production garbage negative token.
#         (b) NaN/Inf logits row (the NVFP4 class) at temp==1 and temp==0:
#             a NaN-dominated row leaves the argmax at p == -1 the same way.
#
# Oracle (both routes): MEMORY SAFETY + a VALIDITY CONTRACT. The int64 output
# is a FRESH device buffer per case. The contract `out_idxs[i] in [0, vocab)`
# is checked for every row, so a garbage/negative/sentinel(-1) index FAILs even
# under `diff` (no sanitizer). poison/initcheck/memcheck additionally catch a
# genuinely unwritten row.
#
# Spec fields (all ints, so the orchestrator generates/shrinks generically):
#   batch_size, vocab_size, k_lo, k_hi (per-row top_k drawn deterministically in
#   [k_lo, k_hi] from a per-row hash of `seed`; used only on the top-k route),
#   temp_milli (temperature * 1000; 0 == greedy), dist (VD_* value dist), seed,
#   gumbel (0 = top-k route, 1 = gumbel/served route), logit_sign (0 = use the
#   dist as-is; 1 = force all logits <= 0, the cleanest temp=0 all--Inf trigger;
#   2 = force all logits >= 0).

from std.random import random_float64, random_ui64, seed as set_seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from std.utils.numerics import inf, nan
from layout import Coord, TileTensor, row_major
from nn.topk import fused_token_sampling_gpu

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    flag,
    flag_int,
    value_dist_name,
)

comptime in_type = DType.float32
comptime out_idx_type = DType.int64
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var batch_size: Int
    var vocab_size: Int
    var k_lo: Int  # per-row top_k lower bound (0 allowed: the S1 trigger)
    var k_hi: Int  # per-row top_k upper bound (< 10: two-stage; else FI)
    var temp_milli: Int  # temperature * 1000 (0 == greedy / argmax)
    var dist: Int  # value-distribution id (see _fuzz: VD_*)
    var seed: Int  # drives the deterministic per-row top_k pattern
    var gumbel: Int  # 0 = top-k route; 1 = gumbel/served route (max_k=-1)
    var logit_sign: Int  # 0 = dist as-is; 1 = all <= 0; 2 = all >= 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "batch_size=",
            self.batch_size,
            " vocab_size=",
            self.vocab_size,
            " k_lo=",
            self.k_lo,
            " k_hi=",
            self.k_hi,
            " temp_milli=",
            self.temp_milli,
            " dist=",
            value_dist_name(self.dist),
            " seed=",
            self.seed,
            " gumbel=",
            self.gumbel,
            " logit_sign=",
            self.logit_sign,
        )


def _row_k(seed_val: Int, row: Int, k_lo: Int, k_hi: Int) -> Int:
    """Deterministic per-row top_k in [k_lo, k_hi].

    Pure function of (seed, row) so `single`-mode re-runs (and the orchestrator's
    shrink/replay) reproduce the exact same per-row pattern. When k_lo == 0 some
    rows draw 0, which is the S1 unwritten-row trigger.
    """
    if k_hi <= k_lo:
        return k_lo
    # splitmix64-ish mix; just needs to be stable and well-spread.
    var h = (
        UInt64(seed_val) * 0x9E3779B97F4A7C15 + UInt64(row) * 0xD1B54A32D192ED03
    )
    h ^= h >> 30
    h *= 0xBF58476D1CE4E5B9
    h ^= h >> 27
    var span = UInt64(k_hi - k_lo + 1)
    return k_lo + Int(h % span)


def gen_specs(n: Int) -> List[CaseSpec]:
    """Generates specs across both routes.

    ~1/2 gumbel/served route (the SERVOPT-1458 path: top_k==-1, temperature
    swept incl. 0, with logit_sign biased to all--negative so a temp=0 divide
    yields an all--Inf row); ~1/2 top-k route (greedy + the S1 per-row-0 sweep).
    """
    var specs = List[CaseSpec]()
    for _ in range(n):
        var batch_size = boundary_int(1, 64, 8)
        var vocab_size = boundary_int(2, 4096, 256)
        var seed_top = Int(random_ui64(0, 1))  # 0 -> gumbel route, 1 -> top-k
        var k_lo: Int
        var k_hi: Int
        var temp_milli: Int
        var dist: Int
        var gumbel: Int
        var logit_sign: Int
        if seed_top == 0:
            # Gumbel / served route. top_k is ignored (max_k forced to -1);
            # temperature is the axis. Bias temp to 0 (the unclamped-divide
            # trigger) but also sweep finite temps.
            gumbel = 1
            k_lo = 1
            k_hi = 1
            var troll = Int(random_ui64(0, 2))
            temp_milli = 0 if troll != 0 else Int(random_ui64(1, 2000))
            # logit_sign: bias to all-negative (1) -> a temp=0 divide makes the
            # whole row -Inf (the clean trigger-(a) p==-1). 3 = all-NaN row
            # (deterministic trigger (b)). Also try all-positive (2) and mixed
            # (0). dist 5 (NaN/Inf specials) is a softer probabilistic (b).
            var sign_roll = Int(random_ui64(0, 4))
            if sign_roll == 0 or sign_roll == 1:
                logit_sign = 1
            elif sign_roll == 2:
                logit_sign = 3
            elif sign_roll == 3:
                logit_sign = 0
            else:
                logit_sign = 2
            dist = 5 if Int(random_ui64(0, 3)) == 0 else 0
        else:
            # Top-k route. Reuse the S1 sweep (per-row 0 allowed).
            gumbel = 0
            logit_sign = 0
            var roll = Int(random_ui64(0, 2))
            if roll == 0:
                k_lo = 1
                k_hi = 1
                temp_milli = 0
            elif roll == 1:
                k_lo = boundary_int(0, 4, 1)
                k_hi = boundary_int(k_lo, 8, 1)
                temp_milli = Int(random_ui64(0, 2000))
            else:
                k_lo = boundary_int(0, 8, 1)
                k_hi = boundary_int(k_lo, 31, 1)
                temp_milli = Int(random_ui64(0, 2000))
            dist = 0 if Int(random_ui64(0, 2)) != 0 else Int(random_ui64(0, 4))
        var the_seed = Int(random_ui64(1, 1_000_000))
        specs.append(
            CaseSpec(
                batch_size,
                vocab_size,
                k_lo,
                k_hi,
                temp_milli,
                dist,
                the_seed,
                gumbel,
                logit_sign,
            )
        )
    return specs^


def _apply_logit_sign[
    dtype: DType
](span: Span[mut=True, Scalar[dtype], _], logit_sign: Int):
    """Shapes the logit row to drive the gumbel argmax triggers.

    0 = no-op (use the dist as-is). 1 = all logits <= 0 (a temp=0 divide then
    makes the whole row -Inf -> the clean trigger-(a) p==-1). 2 = all logits
    >= 0. 3 = force the ENTIRE input to NaN (deterministic trigger (b): a fully
    NaN row, the NVFP4 class). 4 = force the entire input to +Inf (control: an
    all-+Inf row -- +Inf > -Inf is True so argmax should still find p==0).
    """
    var nan_v = nan[dtype]()
    var inf_v = inf[dtype]()
    if logit_sign == 1:
        for i in range(len(span)):
            var v = span[i]
            if v > Scalar[dtype](0):
                span[i] = -v
    elif logit_sign == 2:
        for i in range(len(span)):
            var v = span[i]
            if v < Scalar[dtype](0):
                span[i] = -v
    elif logit_sign == 3:
        for i in range(len(span)):
            span[i] = nan_v
    elif logit_sign == 4:
        for i in range(len(span)):
            span[i] = inf_v


def run_one_case(ctx: DeviceContext, spec: CaseSpec) raises:
    var batch_size = spec.batch_size
    var vocab_size = spec.vocab_size
    var is_gumbel = spec.gumbel == 1

    # Per-row top_k (int64), deterministic from the spec seed. On the gumbel
    # route the kernel ignores k (max_k forced to -1) but we still pass a valid
    # buffer to mirror the production call.
    var k_host = ctx.enqueue_create_host_buffer[out_idx_type](batch_size)
    var max_row_k = 1
    for r in range(batch_size):
        var rk = _row_k(spec.seed, r, spec.k_lo, spec.k_hi)
        k_host[r] = Scalar[out_idx_type](rk)
        if rk > max_row_k:
            max_row_k = rk
    # max_k = -1 routes to gumbel; else the batch-max per-row k (the top-k path).
    var max_k = -1 if is_gumbel else max_row_k

    var k_dev = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    ctx.enqueue_copy(k_dev, k_host)
    var k_tt = TileTensor(k_dev, row_major(batch_size))

    # Per-row temperature (float32). 0 == greedy (the unclamped-divide trigger).
    var temp_val = Float32(spec.temp_milli) / 1000.0
    var temp_host = ctx.enqueue_create_host_buffer[.float32](batch_size)
    for r in range(batch_size):
        temp_host[r] = temp_val
    var temp_dev = ctx.enqueue_create_buffer[.float32](batch_size)
    ctx.enqueue_copy(temp_dev, temp_host)
    var temp_tt = TileTensor(temp_dev, row_major(batch_size))

    # Per-row seed for the sampler RNG (uint64).
    var rng_host = ctx.enqueue_create_host_buffer[.uint64](batch_size)
    for r in range(batch_size):
        rng_host[r] = UInt64(spec.seed + r)
    var rng_dev = ctx.enqueue_create_buffer[.uint64](batch_size)
    ctx.enqueue_copy(rng_dev, rng_host)
    var rng_tt = TileTensor(rng_dev, row_major(batch_size))

    # Logits input [batch_size, vocab_size].
    var in_len = batch_size * vocab_size
    var in_host = ctx.enqueue_create_host_buffer[in_type](in_len)
    fill_by_dist(in_host.as_span(), spec.dist)
    _apply_logit_sign(in_host.as_span(), spec.logit_sign)
    var in_dev = ctx.enqueue_create_buffer[in_type](in_len)
    ctx.enqueue_copy(in_dev, in_host)
    var in_tt = TileTensor(in_dev, row_major(batch_size, vocab_size))

    # FRESH output buffer every case: the oracle. out_idxs is [batch_size, 1].
    # A row the kernel never writes stays at the allocator's fill byte (poison
    # 0xFF -> a huge negative int64) or flags initcheck/memcheck on copy-back;
    # a row written with a sentinel (-1) trips the validity contract below.
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](batch_size)
    var out_tt = TileTensor(out_dev, row_major(batch_size, 1))

    # min_top_p must be 1.0 for the gumbel route to engage (max_k==-1 &&
    # min_top_p==1.0); it is also a valid no-nucleus setting for the top-k route.
    fused_token_sampling_gpu(
        ctx,
        max_k,
        Float32(1.0),
        in_tt,
        out_tt,
        k=k_tt.as_unsafe_any_origin().as_immut(),
        temperature=temp_tt.as_unsafe_any_origin().as_immut(),
        seed=rng_tt.as_unsafe_any_origin().as_immut(),
    )

    # The oracle read: copy the int64 output row(s) back to host.
    var out_host = ctx.enqueue_create_host_buffer[out_idx_type](batch_size)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for r in range(batch_size):
        var tok = Int(out_host[r])
        if tok < 0 or tok >= vocab_size:
            print(
                "FUZZ_CONTRACT_FAIL row=",
                r,
                "token=",
                tok,
                "vocab_size=",
                vocab_size,
                "gumbel=",
                spec.gumbel,
                "temp_milli=",
                spec.temp_milli,
                "logit_sign=",
                spec.logit_sign,
                "row_k=",
                _row_k(spec.seed, r, spec.k_lo, spec.k_hi),
            )
            raise Error("sampled token id out of range [0, vocab_size)")

    _ = k_dev
    _ = temp_dev
    _ = rng_dev
    _ = in_dev
    _ = out_dev
    _ = in_tt
    _ = out_tt


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    set_seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "batch_size=",
                specs[i].batch_size,
                "vocab_size=",
                specs[i].vocab_size,
                "k_lo=",
                specs[i].k_lo,
                "k_hi=",
                specs[i].k_hi,
                "temp_milli=",
                specs[i].temp_milli,
                "dist=",
                specs[i].dist,
                "seed=",
                specs[i].seed,
                "gumbel=",
                specs[i].gumbel,
                "logit_sign=",
                specs[i].logit_sign,
            )
        return

    if mode == "single":
        var batch_size = flag_int(args, "--batch_size", 8)
        var vocab_size = flag_int(args, "--vocab_size", 256)
        var k_lo = flag_int(args, "--k_lo", 1)
        var k_hi = flag_int(args, "--k_hi", 1)
        var temp_milli = flag_int(args, "--temp_milli", 0)
        var dist = flag_int(args, "--dist", 0)
        var spec_seed = flag_int(args, "--seed", the_seed)
        var gumbel = flag_int(args, "--gumbel", 0)
        var logit_sign = flag_int(args, "--logit_sign", 0)
        print(
            "FUZZ_SINGLE batch_size=",
            batch_size,
            "vocab_size=",
            vocab_size,
            "k_lo=",
            k_lo,
            "k_hi=",
            k_hi,
            "temp_milli=",
            temp_milli,
            "dist=",
            dist,
            "seed=",
            spec_seed,
            "gumbel=",
            gumbel,
            "logit_sign=",
            logit_sign,
        )
        with DeviceContext() as ctx:
            run_one_case(
                ctx,
                CaseSpec(
                    batch_size,
                    vocab_size,
                    k_lo,
                    k_hi,
                    temp_milli,
                    dist,
                    spec_seed,
                    gumbel,
                    logit_sign,
                ),
            )
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_topk_sampling seed=", the_seed, "budget=", the_budget, "==="
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i])
    print("=== done:", len(specs), "cases ===")
