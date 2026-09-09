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
"""Tests `topk_topp_sampling_from_prob`'s `emit_dist` output.

With `emit_dist`, the sampler also writes the masked renormalized
distribution it drew from. Speculative decoding subtracts that distribution
to build its rejection residual, and reads the sampled token's probability
out of it.

The reference evaluates the accept predicate directly: a token survives iff
`count(e > e_tok) < k` and `mass(e > e_tok) <= top_p * total`. That is O(d^2),
so the exact cases stay at moderate widths.
"""

from std.math import exp, sqrt
from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major
from std.testing import assert_almost_equal, assert_equal, assert_true

from nn.sampling import topk_topp_sampling_from_prob
from nn.sampling.topk_fi import (
    topk_topp_sampling_from_prob as single_block_sampling_from_prob,
)


def scrambled_logit(row: Int, col: Int) -> Float64:
    """A deterministic, non-monotonic fill spanning a few logit decades."""
    var h = (col * 2654435761 + row * 97) % 1_000_003
    return Float64(h) / 1_000_003.0 * 12.0 - 6.0


def reference_masked_probs(
    logits: List[Float64],
    k: Int,
    top_p: Float64,
    temperature: Float64,
    min_p: Float64 = 0.0,
) -> List[Float64]:
    """Masked, renormalized reference distribution, O(d^2)."""
    var d = len(logits)
    var row_max = logits[0]
    for i in range(1, d):
        row_max = max(row_max, logits[i])

    var inv_temp = 1.0 / max(temperature, 1e-6)
    var e = List[Float64]()
    var total = 0.0
    for i in range(d):
        var v = exp((logits[i] - row_max) * inv_temp)
        e.append(v)
        total += v

    var k_eff = d if (k <= 0 or k > d) else k
    # The top-p budget scales with the unmasked mass; only the working
    # weights below take the min-p mask, matching the kernel.
    var p_eff = top_p.clamp(0.0, 1.0) * total
    if min_p > 0:
        for i in range(d):
            if e[i] < min_p:
                e[i] = 0.0

    # The largest weight that fails the predicate is the cutoff; every token
    # strictly above it survives.
    var cutoff = 0.0
    for i in range(d):
        var count_above = 0
        var mass_above = 0.0
        for j in range(d):
            if e[j] > e[i]:
                count_above += 1
                mass_above += e[j]
        if count_above >= k_eff or mass_above > p_eff:
            cutoff = max(cutoff, e[i])

    var kept_mass = 0.0
    for i in range(d):
        if e[i] > cutoff:
            kept_mass += e[i]

    var probs = List[Float64]()
    for i in range(d):
        probs.append(e[i] / kept_mass if e[i] > cutoff else 0.0)
    return probs^


struct SamplingRun(Movable):
    var tokens: List[Int]
    var dist: List[Float64]

    def __init__(out self, var tokens: List[Int], var dist: List[Float64]):
        self.tokens = tokens^
        self.dist = dist^


def run_sampling[
    dtype: DType = .float32, emit_dist: Bool = True
](
    ctx: DeviceContext,
    logits_host: HostBuffer[dtype],
    rows: Int,
    d: Int,
    *,
    k: Int,
    top_p: Float64,
    temperature: Float64,
    seed_base: UInt64,
    min_p: Float64 = 0.0,
    pass_min_p: Bool = True,
    single_block: Bool = False,
) raises -> SamplingRun:
    var logits_dev = ctx.enqueue_create_buffer[dtype](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)

    var tokens_dev = ctx.enqueue_create_buffer[.int64](rows)
    var dist_len = rows * d if emit_dist else 1
    var dist_dev = ctx.enqueue_create_buffer[.float32](dist_len)

    var temp_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    var min_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    for row in range(rows):
        temp_host[row] = Float32(temperature)
        top_p_host[row] = Float32(top_p)
        top_k_host[row] = Int64(k)
        seed_host[row] = seed_base + UInt64(row)
        min_p_host[row] = Float32(min_p)
    var temp_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](rows)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](rows)
    var min_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(seed_dev, seed_host)
    ctx.enqueue_copy(min_p_dev, min_p_host)

    # Production reaches this kernel through
    # `sampler.fused_token_sampling_with_dist`, which forwards no min-p array,
    # so the unfiltered fast path is only reachable with a null optional here.
    # A buffer of zeros leaves the optional engaged and takes the slow path.
    var min_p_t = (
        TileTensor(min_p_dev, row_major(rows)).as_unsafe_any_origin().as_immut()
    )
    var min_p_arg = Optional(min_p_t)
    if not pass_min_p:
        min_p_arg = None

    # `d` is the top_k default, so a per-row -1 means keep every token. The
    # single-block entry is reachable through the dispatcher only past the
    # shared-memory gate, where the O(d^2) reference is unusable, so exact
    # cases call it directly.
    if single_block:
        single_block_sampling_from_prob[
            from_logits=True, emit_dist=emit_dist, dist_dtype=DType.float32
        ](
            ctx,
            TileTensor(logits_dev, row_major(rows, d)),
            TileTensor(tokens_dev, row_major(rows)),
            d,
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
            min_p=min_p_arg,
            out_dist=TileTensor(
                dist_dev, row_major(rows, d) if emit_dist else row_major(1, 1)
            ).as_unsafe_any_origin(),
        )
    else:
        topk_topp_sampling_from_prob[
            from_logits=True, emit_dist=emit_dist, dist_dtype=DType.float32
        ](
            ctx,
            TileTensor(logits_dev, row_major(rows, d)),
            TileTensor(tokens_dev, row_major(rows)),
            d,
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
            min_p=min_p_arg,
            out_dist=TileTensor(
                dist_dev, row_major(rows, d) if emit_dist else row_major(1, 1)
            ).as_unsafe_any_origin(),
        )

    var tokens_out = ctx.enqueue_create_host_buffer[.int64](rows)
    var dist_out = ctx.enqueue_create_host_buffer[.float32](dist_len)
    ctx.enqueue_copy(tokens_out, tokens_dev)
    ctx.enqueue_copy(dist_out, dist_dev)
    ctx.synchronize()

    var tokens = List[Int]()
    for row in range(rows):
        tokens.append(Int(tokens_out[row]))
    var dist = List[Float64]()
    for i in range(dist_len):
        dist.append(Float64(dist_out[i]))

    _ = logits_dev^
    _ = tokens_dev^
    _ = dist_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = top_k_dev^
    _ = seed_dev^
    _ = min_p_dev^

    return SamplingRun(tokens^, dist^)


def fill_logits[
    dtype: DType
](ctx: DeviceContext, rows: Int, d: Int) raises -> HostBuffer[dtype]:
    var host = ctx.enqueue_create_host_buffer[dtype](rows * d)
    for row in range(rows):
        for col in range(d):
            host[row * d + col] = scrambled_logit(row, col).cast[dtype]()
    return host^


def test_dist_matches_reference[
    dtype: DType = .float32
](
    ctx: DeviceContext,
    rows: Int,
    d: Int,
    *,
    k: Int,
    top_p: Float64,
    temperature: Float64,
    min_p: Float64 = 0.0,
    pass_min_p: Bool = True,
    single_block: Bool = False,
) raises:
    """The emitted distribution equals the reference masked softmax.

    The sampled token must also carry positive mass in it -- that is the
    invariant speculative decoding relies on to read `q` back out.
    """
    var logits_host = fill_logits[dtype](ctx, rows, d)
    var run = run_sampling[dtype](
        ctx,
        logits_host,
        rows,
        d,
        k=k,
        top_p=top_p,
        temperature=temperature,
        seed_base=1234,
        min_p=min_p,
        pass_min_p=pass_min_p,
        single_block=single_block,
    )

    for row in range(rows):
        var logits = List[Float64]()
        for col in range(d):
            logits.append(Float64(logits_host[row * d + col]))
        var expected = reference_masked_probs(
            logits, k, top_p, temperature, min_p
        )

        var dist_sum = 0.0
        for col in range(d):
            var got = run.dist[row * d + col]
            dist_sum += got
            assert_almost_equal(
                got,
                expected[col],
                rtol=1e-3,
                atol=1e-9,
                msg=String(t"row={row} d={d} col={col}: dist mismatch"),
            )
        assert_almost_equal(
            dist_sum,
            1.0,
            rtol=1e-3,
            msg=String(t"row={row} d={d}: dist must sum to 1"),
        )
        assert_true(
            run.dist[row * d + run.tokens[row]] > 0.0,
            msg=String(
                t"row={row} d={d}: sampled token {run.tokens[row]} has no"
                t" mass in the emitted distribution"
            ),
        )


def test_tokens_match_without_dist(
    ctx: DeviceContext, rows: Int, d: Int, *, k: Int, top_p: Float64
) raises:
    """`emit_dist` must not change which token the sampler picks.

    The production sampler runs the same kernel with `emit_dist=False`, so
    the extra output has to be inert.
    """
    var logits_host = fill_logits[.float32](ctx, rows, d)
    var with_dist = run_sampling[.float32, emit_dist=True](
        ctx,
        logits_host,
        rows,
        d,
        k=k,
        top_p=top_p,
        temperature=1.0,
        seed_base=99,
    )
    var without_dist = run_sampling[.float32, emit_dist=False](
        ctx,
        logits_host,
        rows,
        d,
        k=k,
        top_p=top_p,
        temperature=1.0,
        seed_base=99,
    )
    for row in range(rows):
        assert_equal(
            with_dist.tokens[row],
            without_dist.tokens[row],
            msg=String(t"row={row}: emit_dist changed the sampled token"),
        )


def test_empirical_distribution(
    ctx: DeviceContext, rows: Int, d: Int, *, k: Int, top_p: Float64
) raises:
    """Token frequencies over independently seeded rows match the reference.

    Every row carries the same logits, so one launch yields `rows` draws from
    the same distribution.
    """
    var logits_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    for row in range(rows):
        for col in range(d):
            logits_host[row * d + col] = Float32(scrambled_logit(0, col))

    var run = run_sampling[.float32](
        ctx,
        logits_host,
        rows,
        d,
        k=k,
        top_p=top_p,
        temperature=1.0,
        seed_base=7000,
    )

    var logits = List[Float64]()
    for col in range(d):
        logits.append(scrambled_logit(0, col))
    var expected = reference_masked_probs(logits, k, top_p, 1.0)

    var counts = List[Int](length=d, fill=0)
    for row in range(rows):
        counts[run.tokens[row]] += 1

    # Each token's count is Binomial(rows, p); 5 sigma keeps the test stable
    # while still catching a wrong distribution.
    for col in range(d):
        var p = expected[col]
        var observed = Float64(counts[col])
        var mean = p * Float64(rows)
        var sigma = sqrt(mean * (1.0 - p)) + 1.0
        assert_true(
            abs(observed - mean) <= 5.0 * sigma,
            msg=String(
                t"token {col}: observed {observed} vs expected {mean}"
                t" (p={p}, rows={rows})"
            ),
        )
        if p == 0.0:
            assert_equal(
                counts[col],
                0,
                msg=String(t"token {col} is masked out but was sampled"),
            )


def test_empirical_distribution_cluster(
    ctx: DeviceContext,
    *,
    launches: Int,
    rows: Int,
    d: Int,
    k: Int,
    top_p: Float64,
) raises:
    """`test_empirical_distribution` for the cluster kernel.

    That test needs thousands of draws, but a batch that large takes the
    single-block kernel. Here the batch stays small enough to cluster and the
    draws accumulate over many launches, re-seeded per launch.
    """
    var logits_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    for row in range(rows):
        for col in range(d):
            logits_host[row * d + col] = Float32(scrambled_logit(0, col))

    var counts = List[Int](length=d, fill=0)
    for launch in range(launches):
        var run = run_sampling[.float32](
            ctx,
            logits_host,
            rows,
            d,
            k=k,
            top_p=top_p,
            temperature=1.0,
            seed_base=9000 + UInt64(launch * rows),
        )
        for row in range(rows):
            counts[run.tokens[row]] += 1

    var logits = List[Float64]()
    for col in range(d):
        logits.append(scrambled_logit(0, col))
    var expected = reference_masked_probs(logits, k, top_p, 1.0)

    var n = launches * rows
    for col in range(d):
        var p = expected[col]
        var observed = Float64(counts[col])
        var mean = p * Float64(n)
        var sigma = sqrt(mean * (1.0 - p)) + 1.0
        assert_true(
            abs(observed - mean) <= 5.0 * sigma,
            msg=String(
                t"token {col}: observed {observed} vs expected {mean}"
                t" (p={p}, n={n})"
            ),
        )
        if p == 0.0:
            assert_equal(
                counts[col],
                0,
                msg=String(t"token {col} is masked out but was sampled"),
            )


def main() raises:
    with DeviceContext() as ctx:
        test_dist_matches_reference(
            ctx, rows=4, d=1024, k=-1, top_p=1.0, temperature=1.0
        )
        test_dist_matches_reference(
            ctx, rows=4, d=257, k=8, top_p=1.0, temperature=1.0
        )
        test_dist_matches_reference(
            ctx, rows=4, d=1024, k=-1, top_p=0.9, temperature=1.0
        )
        test_dist_matches_reference(
            ctx, rows=4, d=1024, k=64, top_p=0.8, temperature=0.7
        )
        # Greedy rows: temperature 0 is clamped and collapses to the argmax.
        test_dist_matches_reference(
            ctx, rows=2, d=256, k=-1, top_p=1.0, temperature=0.0
        )
        test_dist_matches_reference[.bfloat16](
            ctx, rows=2, d=512, k=32, top_p=0.95, temperature=1.0
        )
        # A batch wider than the SM count: the grid runs multiple waves of
        # clusters.
        test_dist_matches_reference(
            ctx, rows=160, d=512, k=16, top_p=0.9, temperature=1.0
        )
        # A vocabulary whose per-CTA slice does not fit the shared-memory
        # budget: the launcher falls back to the single-block kernel and its
        # emit_dist tail, which are otherwise unexercised on cluster devices.
        test_tokens_match_without_dist(ctx, rows=2, d=249856, k=40, top_p=0.95)
        # Exercise two- and four-block row groups.
        test_tokens_match_without_dist(ctx, rows=100, d=16384, k=40, top_p=0.95)
        test_tokens_match_without_dist(ctx, rows=50, d=32768, k=40, top_p=0.95)
        # Vocabularies narrower than one block: under a cluster most CTAs own
        # no elements and contribute only reduction identities -- the shape
        # the rejection sampler reaches with its unit-test vocabularies.
        test_dist_matches_reference(
            ctx, rows=4, d=6, k=-1, top_p=0.6, temperature=1.0
        )
        test_dist_matches_reference(
            ctx, rows=4, d=8, k=4, top_p=0.9, temperature=1.0
        )
        # The unfiltered fast path: top-k disabled, top_p saturated and no
        # min-p array, so every positive token survives and the cutoff search
        # is skipped entirely. This is the shape
        # `sampler.fused_token_sampling_with_dist` launches, and it is only
        # reachable with a null min-p optional -- a zero-filled buffer keeps
        # the optional engaged and takes the search path instead.
        test_dist_matches_reference(
            ctx,
            rows=4,
            d=1024,
            k=-1,
            top_p=1.0,
            temperature=1.0,
            pass_min_p=False,
        )
        test_dist_matches_reference(
            ctx,
            rows=4,
            d=1024,
            k=-1,
            top_p=1.0,
            temperature=1.0,
            pass_min_p=False,
            single_block=True,
        )
        # A null min-p array with the constraints still active must keep
        # taking the search path and agree with the reference.
        test_dist_matches_reference(
            ctx,
            rows=4,
            d=1024,
            k=-1,
            top_p=0.9,
            temperature=1.0,
            pass_min_p=False,
        )
        # Min-p masks weight out of the working distribution while the
        # sampling budget stays the unmasked mass, so the emitted row must
        # renormalize by the masked kept mass -- on both launch shapes, and
        # in particular when the whole row passes top-p at the first trial
        # (the regression the fuzzer found: the row summed to less than 1).
        test_dist_matches_reference(
            ctx, rows=4, d=1000, k=0, top_p=0.95, temperature=1.0, min_p=0.05
        )
        test_dist_matches_reference(
            ctx, rows=3, d=3, k=0, top_p=0.951, temperature=0.5, min_p=0.149
        )
        test_dist_matches_reference(
            ctx,
            rows=4,
            d=1000,
            k=0,
            top_p=0.95,
            temperature=1.0,
            min_p=0.05,
            single_block=True,
        )
        test_dist_matches_reference(
            ctx,
            rows=3,
            d=3,
            k=0,
            top_p=0.951,
            temperature=0.5,
            min_p=0.149,
            single_block=True,
        )
        test_dist_matches_reference(
            ctx,
            rows=4,
            d=512,
            k=40,
            top_p=0.9,
            temperature=0.7,
            min_p=0.02,
            single_block=True,
        )

        test_tokens_match_without_dist(ctx, rows=8, d=1024, k=32, top_p=0.9)
        # Real Llama-3.1-8B vocabulary width.
        test_tokens_match_without_dist(ctx, rows=2, d=128256, k=40, top_p=0.95)
        # A vocabulary in the two-CTA window: too wide for one CTA's staging
        # budget, narrow enough to split across two.
        test_tokens_match_without_dist(ctx, rows=2, d=98304, k=40, top_p=0.95)

        test_empirical_distribution(ctx, rows=8192, d=64, k=8, top_p=1.0)
        test_empirical_distribution(ctx, rows=8192, d=128, k=-1, top_p=0.9)
        test_empirical_distribution_cluster(
            ctx, launches=512, rows=8, d=64, k=8, top_p=1.0
        )
        test_empirical_distribution_cluster(
            ctx, launches=512, rows=8, d=128, k=-1, top_p=0.9
        )
