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
"""The indexer top-k at multi-turn context lengths, over four distributions.

A conversation of 14k initial context plus twenty 7k increments gives
`N = 14336 + 7168 * i` for `i` in 0..20, so `N` runs 14336..157696. The indexer
top-k runs once per query token, so `rows` is a decode batch or a prefill
chunk's token count. All but the shortest of these `N` are past the widest
register-resident payload, so they land on the streaming select, and `rows`
decides only whether it takes the prefetching variant (below the SM count) or
the occupancy one (at or above it). The first rung is the exception, and only
for the rank-free contracts, which reach further before they have to stream --
so it is also where the two designs are closest.

    decode:  rows=48 (batch),        N=14336..157696, K=2048
    prefill: rows=256+ (one chunk),  N=14336..157696, K=2048

`dist` selects the score distribution, because the select's cost is set by how
many columns share the threshold's coarse bin and that is a property of the
distribution, not of `N`:

    q17     ~17 discrete levels -- the tie plateaus fp8-quantized indexer
            scores collapse to. The threshold bin is a wide plateau.
    normal  standard normal -- what an attention logit actually looks like,
            being a sum of many products. Bins cluster near the mode.
    uniform uniform on [-1, 1] -- bins evenly occupied.
    narrow  every value inside one coarse bin (1 + u/32) with real spread
            inside it, so the threshold bin holds the whole row. Adversarial.

`mode` picks the output contract, and all four combinations of the kernel's two
relaxations are reachable so that the effect of each can be measured separately
rather than asserted:

    ord       descending score, ascending column -- the default.
    ord_nd    ordered, `deterministic=False`. Documented to be the same thing
              as `ord`, since an ordered output is reproducible whatever the
              flag says; here to hold that claim to a measurement.
    unord     the same `K` columns, order unspecified but reproducible.
    unord_nd  the same `K` columns, neither ordered nor reproducible. Above the
              resident width it is the only one that skips the rank -- see
              `_histsel_topk_kernel` for why reproducibility costs the rank
              there.
"""

from std.memory import bitcast

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import TileTensor, row_major

from nn.topk_bitonic import persistent_topk_block_split


def _get_run_name(
    rows: Int, N: Int, K: Int, dist: String, mode: String
) -> String:
    return String(
        "topk_bitonic_split : rows=",
        rows,
        ", N=",
        N,
        ", K=",
        K,
        ", dist=",
        dist,
        ", mode=",
        mode,
    )


@always_inline
def _mix64(x: UInt64) -> UInt64:
    """SplitMix64's finalizer."""
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def _u01(i: UInt64) -> Float64:
    """A uniform on [0, 1) keyed by `i` alone.

    Keyed rather than sequential so a harness in another language can generate the
    *same* array without sharing an RNG: different languages have different
    generators, and a distribution that merely matches in family is not the same
    input. The select's cost is set by how many columns share the threshold's
    coarse bin, which is a property of the actual values, so an honest comparison
    needs the values themselves to be equal -- hence the checksum this bench
    prints, which a comparing harness can be held to.
    """
    var h = _mix64((i + 1) * 0x9E3779B97F4A7C15)
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


@always_inline
def _u_val(r: Int, c: Int, N: Int, k: Int) -> Float64:
    return _u01(((UInt64(r) * UInt64(N) + UInt64(c)) << 2) + UInt64(k))


@always_inline
def _u_len(r: Int) -> Float64:
    return _u01(0x8000000000000000 + UInt64(r))


# Irwin-Hall rather than Box-Muller: Box-Muller needs `log` and `cos`, and two
# languages' libms need not agree in the last bit, which is enough to make two
# harnesses generate different arrays from the same key. Sums and a literal scale
# are exact IEEE operations, so this is reproducible anywhere. Variance of the
# terms below is 1/2, hence the scale.
comptime _IH_TERMS: Int = 6
comptime _IH_SCALE: Float64 = 1.4142135623730951


def _sample(dist: String, r: Int, c: Int, N: Int) -> Float32:
    """One score from `dist`. See the module docstring for what each one is for.
    """
    var u = _u_val(r, c, N, 0)
    if dist == "q17":
        return Float32(Int(u * 17.0)) / 16.0
    if dist == "uniform":
        return Float32(u * 2.0 - 1.0)
    if dist == "narrow":
        return Float32(1.0 + u / 32.0)
    # One index computation for the whole sum: only the term number varies.
    var base = (UInt64(r) * UInt64(N) + UInt64(c)) << 2
    var acc = Float64(0.0)
    for k in range(_IH_TERMS):
        acc += _u01(base + UInt64(k))
    return Float32((acc - Float64(_IH_TERMS) / 2.0) * _IH_SCALE)


def execute_topk_bitonic[
    ordered: Bool, deterministic: Bool
](
    ctx: DeviceContext,
    mut m: Bench,
    rows: Int,
    N: Int,
    K: Int,
    dist: String,
    mode: String,
) raises:
    """Benches `persistent_topk_block_split` on scores shaped like the DSA
    indexer's: a random-length valid prefix per row (varying `num_keys`
    within a causal chunk), an `-inf` masked suffix, and valid values drawn
    from `dist`.
    """
    var scores_buf = ctx.enqueue_create_buffer[.float32](rows * N)
    var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)

    var csum = UInt64(0xCBF29CE484222325)
    with scores_buf.map_to_host() as h:
        for r in range(rows):
            var num_keys = N - Int(_u_len(r) * Float64(N) / 8.0)
            for c in range(N):
                var v: Float32
                if c < num_keys:
                    v = _sample(dist, r, c, N)
                else:
                    v = Float32(-3.0e38)  # min_or_neg_inf sentinel
                h[r * N + c] = Float32(v)
                csum = (csum ^ UInt64(bitcast[.uint32, 1](v))) * 0x100000001B3
    print("input_checksum=", hex(csum), sep="")

    var scores_t = TileTensor(scores_buf, row_major(rows, N))
    var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
    ctx.synchronize()

    @always_inline
    def kernel_launch(c: DeviceContext) raises {mut idxs_t, imm}:
        persistent_topk_block_split[
            ordered=ordered, deterministic=deterministic
        ](
            c,
            rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            N,
            K,
            rows,
        )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func, BenchId(_get_run_name(rows, N, K, dist, mode)), []
    )

    _ = scores_buf
    _ = idxs_buf


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 157696)
    var K = arg_parse("K", 2048)
    var dist = arg_parse("dist", String("q17"))
    var mode = arg_parse("mode", String("ord"))

    # An unrecognized mode or distribution is refused rather than falling back to
    # a default. A sweep that asks for a contract this binary does not have would
    # otherwise measure the default four times over and report that the flags
    # change nothing; a misspelled distribution would report the fallback's
    # numbers under the label as typed, which is how the adversarial one comes to
    # look benign. Both are wrong answers that look like findings.
    if dist not in ["q17", "normal", "uniform", "narrow"]:
        raise Error("unknown dist: ", dist)
    if mode not in ["ord", "ord_nd", "unord", "unord_nd"]:
        raise Error("unknown mode: ", mode)

    var m = Bench()
    with DeviceContext() as ctx:
        if mode == "unord_nd":
            execute_topk_bitonic[False, False](ctx, m, rows, N, K, dist, mode)
        elif mode == "unord":
            execute_topk_bitonic[False, True](ctx, m, rows, N, K, dist, mode)
        elif mode == "ord_nd":
            execute_topk_bitonic[True, False](ctx, m, rows, N, K, dist, mode)
        else:
            execute_topk_bitonic[True, True](ctx, m, rows, N, K, dist, mode)

    m.dump_report()
