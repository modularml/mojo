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
"""Tests that `top_p == 1.0` leaves the nucleus constraint disabled."""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import exp
from std.testing import assert_almost_equal, assert_equal

from nn.sampling import topk_topp_masked_probs, topk_topp_sampling_from_prob


def _u01(col: Int) -> Float64:
    var h = UInt64(col) * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9
    h ^= h >> 30
    h *= 0xBF58476D1CE4E5B9
    h ^= h >> 27
    h *= 0x94D049BB133111EB
    h ^= h >> 31
    return Float64(h >> 11) * (1.0 / Float64(UInt64(1) << 53))


comptime _HEAD = 5


def ulp_thin_tail_logit(row: Int, col: Int) -> Float64:
    if col < _HEAD:
        return -Float64(col + row) / Float64(_HEAD)
    return -30.0 - 0.5 * _u01(col + 7919 * row)


def expected_survivors(
    d: Int, k: Int, logits: List[Float64], min_p: Float64
) -> List[Bool]:
    var row_max = logits[0]
    for i in range(1, d):
        row_max = max(row_max, logits[i])

    var e = List[Float64]()
    for i in range(d):
        var v = exp(logits[i] - row_max)
        e.append(0.0 if v < min_p else v)

    var sorted_e = e.copy()

    def _greater_than(lhs: Float64, rhs: Float64) -> Bool:
        return lhs > rhs

    sort(sorted_e, _greater_than)
    var kth = sorted_e[k - 1]

    var keep = List[Bool]()
    for i in range(d):
        keep.append(e[i] > 0.0 and e[i] >= kth)
    return keep^


def check_row(
    d: Int,
    k: Int,
    row: Int,
    got: List[Float64],
    logits: List[Float64],
    label: String,
    min_p: Float64 = 0.0,
) raises:
    var keep = expected_survivors(d, k, logits, min_p)
    var n_kept = 0
    var n_expected = 0
    for col in range(d):
        if keep[col]:
            n_expected += 1
        if got[col] != 0.0:
            n_kept += 1
    assert_equal(
        n_kept,
        n_expected,
        msg=String(
            t"{label} d={d} k={k} row={row}: top_p=1.0 must keep exactly the"
            t" top-k tokens"
        ),
    )
    var total = 0.0
    for col in range(d):
        assert_equal(
            got[col] != 0.0,
            keep[col],
            msg=String(t"{label} d={d} k={k} row={row} col={col}: wrong token"),
        )
        total += got[col]
    assert_almost_equal(
        total,
        1.0,
        rtol=1e-3,
        msg=String(t"{label} d={d} k={k} row={row}: probs must sum to 1"),
    )


def make_logits(ctx: DeviceContext, rows: Int, d: Int) raises -> List[Float64]:
    var out = List[Float64]()
    for row in range(rows):
        for col in range(d):
            out.append(Float64(Float32(ulp_thin_tail_logit(row, col))))
    return out^


def run_masked_probs(
    ctx: DeviceContext, rows: Int, d: Int, k: Int, top_p: Float32 = 1.0
) raises:
    var logits = make_logits(ctx, rows, d)
    var logits_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    for i in range(rows * d):
        logits_host[i] = Float32(logits[i])
    var logits_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)
    var probs_dev = ctx.enqueue_create_buffer[.float32](rows * d)

    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    for row in range(rows):
        top_p_host[row] = top_p
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

    # The thin tail falls outside any nucleus below 1.
    var k_eff = k if top_p >= 1.0 else _HEAD
    for row in range(rows):
        var got = List[Float64]()
        var row_logits = List[Float64]()
        for col in range(d):
            got.append(Float64(probs_host[row * d + col]))
            row_logits.append(logits[row * d + col])
        check_row(d, k_eff, row, got, row_logits, "masked_probs")

    _ = logits_dev^
    _ = probs_dev^
    _ = top_p_dev^
    _ = top_k_dev^


def run_sampling_dist(
    ctx: DeviceContext, rows: Int, d: Int, k: Int, min_p: Float64 = 0.0
) raises:
    var logits = make_logits(ctx, rows, d)
    var logits_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    for i in range(rows * d):
        logits_host[i] = Float32(logits[i])
    var logits_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    ctx.enqueue_copy(logits_dev, logits_host)
    var dist_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    var tokens_dev = ctx.enqueue_create_buffer[.int64](rows)

    var top_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    var top_k_host = ctx.enqueue_create_host_buffer[.int64](rows)
    var seed_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    var min_p_host = ctx.enqueue_create_host_buffer[.float32](rows)
    for row in range(rows):
        top_p_host[row] = 1.0
        top_k_host[row] = Int64(k)
        seed_host[row] = UInt64(row) + 12345
        min_p_host[row] = Float32(min_p)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](rows)
    var seed_dev = ctx.enqueue_create_buffer[.uint64](rows)
    var min_p_dev = ctx.enqueue_create_buffer[.float32](rows)
    ctx.enqueue_copy(top_p_dev, top_p_host)
    ctx.enqueue_copy(top_k_dev, top_k_host)
    ctx.enqueue_copy(seed_dev, seed_host)
    ctx.enqueue_copy(min_p_dev, min_p_host)

    # Exercise both the masked and unmasked paths.
    var min_p_t = (
        TileTensor(min_p_dev, row_major(rows)).as_unsafe_any_origin().as_immut()
    )
    var min_p_arg = Optional(min_p_t)
    if min_p == 0.0:
        min_p_arg = None

    topk_topp_sampling_from_prob[
        from_logits=True, emit_dist=True, dist_dtype=DType.float32
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
        min_p=min_p_arg,
        out_dist=TileTensor(
            dist_dev, row_major(rows, d)
        ).as_unsafe_any_origin(),
    )

    var dist_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    var tokens_host = ctx.enqueue_create_host_buffer[.int64](rows)
    ctx.enqueue_copy(dist_host, dist_dev)
    ctx.enqueue_copy(tokens_host, tokens_dev)
    ctx.synchronize()

    for row in range(rows):
        var got = List[Float64]()
        var row_logits = List[Float64]()
        for col in range(d):
            got.append(Float64(dist_host[row * d + col]))
            row_logits.append(logits[row * d + col])
        check_row(d, k, row, got, row_logits, "sampling_dist", min_p)
        assert_equal(
            got[Int(tokens_host[row])] != 0.0,
            True,
            msg=String(
                t"sampling_dist d={d} k={k} row={row}: token off-nucleus"
            ),
        )

    _ = logits_dev^
    _ = dist_dev^
    _ = tokens_dev^
    _ = top_p_dev^
    _ = top_k_dev^
    _ = seed_dev^
    _ = min_p_dev^


def main() raises:
    with DeviceContext() as ctx:
        for d in [4096, 8192, 32768]:
            for k in [1, _HEAD, 50, d - 1]:
                run_masked_probs(ctx, rows=2, d=d, k=k)
                run_sampling_dist(ctx, rows=2, d=d, k=k)

        for p in [Float32(0.9), Float32(0.9999), Float32(0.99999994)]:
            run_masked_probs(ctx, rows=2, d=4096, k=50, top_p=p)

        run_sampling_dist(ctx, rows=2, d=4096, k=50, min_p=0.05)
        run_sampling_dist(ctx, rows=2, d=4096, k=4095, min_p=0.05)
