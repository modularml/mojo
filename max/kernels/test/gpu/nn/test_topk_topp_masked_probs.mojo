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
"""Correctness tests for `nn.sampling.topk_topp_masked_probs`.

The kernel writes, per row, the joint top-k/top-p masked renormalized
softmax: a token survives iff its weight `e_i = exp((logit_i - row_max) /
temperature)` clears the constraint-set cutoff, with probability
`e_i / kept_mass`, and every other slot is zero. Speculative decoding
verification reads target probabilities and builds its rejection residual
straight out of this tensor.

The reference here evaluates the defining predicate directly -- a token
survives iff `count(e > e_tok) < k` and `mass(e > e_tok) <= top_p * total`
-- by scanning the whole row for every candidate. That is O(d^2), so the
exact-reference cases stay at moderate widths; vocabulary-scale rows are
covered by the small-k reference (k passes of max extraction, O(d*k)) and
by the mask-disabled identity (the row is its plain softmax).
"""

from std.math import exp
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.testing import assert_almost_equal, assert_equal

from nn.sampling import topk_topp_masked_probs


def scrambled_logit(row: Int, col: Int, d: Int) -> Float64:
    """A deterministic, non-monotonic fill spanning a few logit decades."""
    var h = (col * 2654435761 + row * 97) % 1_000_003
    return Float64(h) / 1_000_003.0 * 12.0 - 6.0


def ascending_logit(row: Int, col: Int, d: Int) -> Float64:
    """Strictly increasing in `col`, so the descending order is known."""
    return Float64(col) / Float64(d) * 10.0 + Float64(row) * 0.25


def reference_masked_probs(
    logits: List[Float64], k: Int, top_p: Float64, temperature: Float64
) -> List[Float64]:
    """Evaluates the joint constraint predicate for every token, O(d^2)."""
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

    var k_eff = k
    if k_eff <= 0 or k_eff > d:
        k_eff = d
    var p_eff = top_p.clamp(0.0, 1.0) * total

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


def reference_masked_probs_topk_only(
    logits: List[Float64], k: Int, temperature: Float64
) -> List[Float64]:
    """Top-k-only reference via k passes of max extraction, O(d*k)."""
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

    var probs = List[Float64]()
    var k_eff = k
    if k_eff <= 0 or k_eff > d:
        for i in range(d):
            probs.append(e[i] / total)
        return probs^

    # Value of the k-th largest element; the surviving set keeps every token
    # at or above it (ties included, matching the kernel). The test fillers
    # produce distinct values, so extracting k distinct values is the k-th
    # element here.
    var bound = Float64.MAX
    var kth = 0.0
    for _ in range(k_eff):
        var best = -1.0
        for i in range(d):
            if e[i] < bound and e[i] > best:
                best = e[i]
        kth = best
        bound = best

    var kept_mass = 0.0
    for i in range(d):
        if e[i] >= kth:
            kept_mass += e[i]
    for i in range(d):
        probs.append(e[i] / kept_mass if e[i] >= kth else 0.0)
    return probs^


def run_probs[
    dtype: DType = .float32
](
    ctx: DeviceContext,
    rows: Int,
    d: Int,
    *,
    filler: def(Int, Int, Int) thin -> Float64,
    k: Int,
    top_p: Float64,
    temperature: Float64,
    exact_reference: Bool,
) raises:
    var logits_host = ctx.enqueue_create_host_buffer[dtype](rows * d)
    for row in range(rows):
        for col in range(d):
            logits_host[row * d + col] = filler(row, col, d).cast[dtype]()

    var logits_dev = ctx.enqueue_create_buffer[dtype](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)

    var probs_dev = ctx.enqueue_create_buffer[.float32](rows * d)

    var temp_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    for row in range(rows):
        temp_host[row] = Float32(temperature)
        top_p_host[row] = Float32(top_p)
        top_k_host[row] = Int64(k)
    var temp_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](rows)
    ctx.enqueue_copy(temp_dev, temp_host)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)

    topk_topp_masked_probs(
        ctx,
        TileTensor(logits_dev, row_major(rows, d)),
        TileTensor(probs_dev, row_major(rows, d)).as_unsafe_any_origin(),
        top_k_val=d,
        top_p_val=1.0,
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

    var probs_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    ctx.enqueue_copy(probs_host, probs_dev)
    ctx.synchronize()

    for row in range(rows):
        var logits = List[Float64]()
        for col in range(d):
            logits.append(Float64(logits_host[row * d + col]))

        var expected: List[Float64]
        if exact_reference:
            expected = reference_masked_probs(logits, k, top_p, temperature)
        else:
            expected = reference_masked_probs_topk_only(logits, k, temperature)

        var prob_sum = 0.0
        for col in range(d):
            var got = Float64(probs_host[row * d + col])
            prob_sum += got
            assert_almost_equal(
                got,
                expected[col],
                rtol=1e-3,
                atol=1e-9,
                msg=String(t"row={row} d={d} k={k} col={col}: prob mismatch"),
            )
        assert_almost_equal(
            prob_sum,
            1.0,
            rtol=1e-3,
            msg=String(t"row={row} d={d} k={k}: probs must sum to 1"),
        )

    _ = logits_dev^
    _ = probs_dev^
    _ = temp_dev^
    _ = top_p_dev^
    _ = top_k_dev^


def test_empty_batch(ctx: DeviceContext, d: Int) raises:
    """Zero rows must be a no-op, not a zero-sized grid launch.

    Speculative decoding calls this with no rows on every step that has no
    drafts to verify (any prefill-only step), so the empty batch is a normal
    input rather than an edge case.
    """
    var logits = ctx.enqueue_create_buffer[.float32](0)
    var probs = ctx.enqueue_create_buffer[.float32](0)

    topk_topp_masked_probs(
        ctx,
        TileTensor(logits, row_major(0, d)),
        TileTensor(probs, row_major(0, d)).as_unsafe_any_origin(),
        top_k_val=d,
        top_p_val=1.0,
    )
    ctx.synchronize()

    _ = logits^
    _ = probs^


def test_narrow_cluster_brackets(
    ctx: DeviceContext, rows: Int, d: Int, k: Int
) raises:
    var logits_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    for row in range(rows):
        for col in range(d):
            logits_host[row * d + col] = Float32(-Float64(col + row) * 8.94e-8)
    var logits_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)
    var probs_dev = ctx.enqueue_create_buffer[.float32](rows * d)

    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    for row in range(rows):
        top_p_host[row] = 1.0
        top_k_host[row] = Int64(k)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](rows)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)

    topk_topp_masked_probs(
        ctx,
        TileTensor(logits_dev, row_major(rows, d)),
        TileTensor(probs_dev, row_major(rows, d)).as_unsafe_any_origin(),
        top_k_val=d,
        top_p_val=1.0,
        top_k_arr=TileTensor(top_k_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
        top_p_arr=TileTensor(top_p_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
    )

    var probs_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    ctx.enqueue_copy(probs_host, probs_dev)
    ctx.synchronize()

    for row in range(rows):
        for col in range(d):
            assert_equal(
                probs_host[row * d + col] != 0.0,
                col < k,
                msg=String(t"narrow bracket d={d} k={k} row={row} col={col}"),
            )

    _ = logits_dev^
    _ = probs_dev^
    _ = top_p_dev^
    _ = top_k_dev^


def main() raises:
    with DeviceContext() as ctx:
        for k in [32, 45, 51, 61, 63]:
            test_narrow_cluster_brackets(ctx, rows=2, d=4096, k=k)

        test_empty_batch(ctx, d=1024)
        # Vocabularies far narrower than one cluster's thread count. At these
        # widths a row does not even fill one block, so under a cluster most
        # CTAs own no elements at all and contribute only reduction
        # identities -- the shape `rejection_sampler.py` reaches with
        # `vocab_size` 6 and 8.
        run_probs(
            ctx,
            rows=4,
            d=6,
            filler=scrambled_logit,
            k=-1,
            top_p=0.9,
            temperature=1.0,
            exact_reference=True,
        )
        run_probs(
            ctx,
            rows=4,
            d=8,
            filler=scrambled_logit,
            k=3,
            top_p=1.0,
            temperature=1.0,
            exact_reference=True,
        )
        run_probs(
            ctx,
            rows=6,
            d=8,
            filler=ascending_logit,
            k=-1,
            top_p=0.5,
            temperature=1.0,
            exact_reference=True,
        )
        # Mask disabled: the row is its plain softmax.
        run_probs(
            ctx,
            rows=4,
            d=1024,
            filler=scrambled_logit,
            k=-1,
            top_p=1.0,
            temperature=1.0,
            exact_reference=True,
        )
        # Top-k only, top-p only, and joint, at exact-reference widths.
        run_probs(
            ctx,
            rows=4,
            d=257,
            filler=scrambled_logit,
            k=8,
            top_p=1.0,
            temperature=1.0,
            exact_reference=True,
        )
        run_probs(
            ctx,
            rows=4,
            d=1024,
            filler=scrambled_logit,
            k=-1,
            top_p=0.9,
            temperature=1.0,
            exact_reference=True,
        )
        run_probs(
            ctx,
            rows=3,
            d=1024,
            filler=scrambled_logit,
            k=64,
            top_p=0.8,
            temperature=0.7,
            exact_reference=True,
        )
        # k >= d is equivalent to disabling top-k.
        run_probs(
            ctx,
            rows=2,
            d=512,
            filler=ascending_logit,
            k=512,
            top_p=1.0,
            temperature=1.0,
            exact_reference=True,
        )
        # Greedy rows: temperature 0 is clamped, collapsing to the argmax.
        run_probs(
            ctx,
            rows=2,
            d=256,
            filler=ascending_logit,
            k=-1,
            top_p=1.0,
            temperature=0.0,
            exact_reference=True,
        )
        # Real Llama-3.1-8B vocabulary width.
        run_probs(
            ctx,
            rows=2,
            d=128256,
            filler=scrambled_logit,
            k=-1,
            top_p=1.0,
            temperature=1.0,
            exact_reference=False,
        )
        run_probs(
            ctx,
            rows=2,
            d=128256,
            filler=scrambled_logit,
            k=40,
            top_p=1.0,
            temperature=1.0,
            exact_reference=False,
        )
        # A batch wider than the SM count: the grid runs multiple waves of
        # clusters.
        run_probs(
            ctx,
            rows=200,
            d=257,
            filler=scrambled_logit,
            k=-1,
            top_p=0.9,
            temperature=1.0,
            exact_reference=True,
        )
        run_probs(
            ctx,
            rows=200,
            d=8,
            filler=ascending_logit,
            k=-1,
            top_p=0.5,
            temperature=1.0,
            exact_reference=True,
        )
        # A vocabulary in the two-CTA window: too wide for one CTA's staging
        # budget, narrow enough to split across two.
        run_probs(
            ctx,
            rows=2,
            d=98304,
            filler=scrambled_logit,
            k=40,
            top_p=0.95,
            temperature=1.0,
            exact_reference=False,
        )
        # A vocabulary whose per-CTA slice does not fit the shared-memory
        # budget: the launcher falls back to the single-block kernel, which
        # is otherwise unexercised on cluster devices.
        run_probs(
            ctx,
            rows=2,
            d=249856,
            filler=scrambled_logit,
            k=40,
            top_p=1.0,
            temperature=1.0,
            exact_reference=False,
        )
        # Exercise two- and four-block row groups.
        run_probs(
            ctx,
            rows=100,
            d=16384,
            filler=scrambled_logit,
            k=40,
            top_p=0.95,
            temperature=1.0,
            exact_reference=False,
        )
        run_probs(
            ctx,
            rows=50,
            d=32768,
            filler=scrambled_logit,
            k=40,
            top_p=0.95,
            temperature=1.0,
            exact_reference=False,
        )
