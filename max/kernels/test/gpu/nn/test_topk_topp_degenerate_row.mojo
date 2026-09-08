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

"""Memory-safety gate: a degenerate row must not produce an out-of-range index.

A row whose fused softmax has no element above the initial pivot (`low == 0`)
leaves the sampler's dual-pivot search with no candidate. The fallback index
must still be in `[0, vocab)`: it feeds `load_dist` as a row offset, so an
out-of-range value is an illegal device access (observed in production as
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`), not merely a wrong token.

Degenerate rows are reachable from any non-finite logit: `row_max` becomes
`+inf`/`NaN`, so every `exp((logit - row_max) * inv_temp)` is `NaN`, and `NaN`
compares false against every pivot.

Both samplers share the defect, so both are covered: the top-k+top-p kernel from
logits, and the top-k-only kernel from probabilities (where a degenerate row is
an all-zero or all-`NaN` distribution).

`poison_lds_kernel` runs first so the sampler's shared-memory segment holds a
huge sentinel rather than a benign zero -- without it an uninitialized fallback
slot reads back as 0 and the bug hides.
"""

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx
from std.memory import AddressSpace
from std.memory import unsafe_stack_allocation
from layout import Coord, TileTensor, row_major
from nn.sampling import topk_sampling_from_prob, topk_topp_sampling_from_prob
from std.testing import assert_true
from std.utils import IndexList
from std.utils.numerics import inf, nan, neg_inf

comptime DTYPE = DType.float32
comptime IDX = DType.int64
comptime BLOCK = 1024

comptime ROW_WELL_FORMED = 0
comptime ROW_ALL_NEG_INF = 1
comptime ROW_ONE_POS_INF = 2
comptime ROW_ALL_NAN = 3
comptime ROW_ALL_ZERO = 4


def poison_lds_kernel[nbytes: Int]():
    var p = unsafe_stack_allocation[nbytes // 8, Int64, address_space=.SHARED]()
    for i in range(thread_idx.x, nbytes // 8, BLOCK):
        p[i] = Int64(0x0BADF00DDEADBEEF)
    barrier()
    # Keep the stores from being folded away.
    if thread_idx.x == 0 and block_idx.x == 0xFFFFFFFF:
        print(p[0])


def check_degenerate_row(
    ctx: DeviceContext,
    name: String,
    n: Int,
    row_kind: List[Int],
    top_p: Float32,
) raises:
    var batch_size = len(row_kind)
    var in_shape = IndexList[2](batch_size, n)
    var in_layout = row_major(Coord(in_shape))

    var d_in = ctx.enqueue_create_buffer[DTYPE](in_shape.flattened_length())
    var d_out = ctx.enqueue_create_buffer[IDX](batch_size)
    var seed_buf = ctx.enqueue_create_buffer[.uint64](batch_size)
    var temp_buf = ctx.enqueue_create_buffer[.float32](batch_size)

    with d_in.map_to_host() as h_in:
        var t = TileTensor(h_in, in_layout)
        for b in range(batch_size):
            var kind = row_kind[b]
            for i in range(n):
                if kind == ROW_ALL_NEG_INF:
                    t[b, i] = neg_inf[DTYPE]()
                elif kind == ROW_ALL_NAN:
                    t[b, i] = nan[DTYPE]()
                else:
                    t[b, i] = Scalar[DTYPE](0.001 * Float32(i % 97))
            if kind == ROW_ONE_POS_INF:
                t[b, n // 2] = inf[DTYPE]()

    with d_out.map_to_host() as h_out:
        for b in range(batch_size):
            h_out[b] = Scalar[IDX](-7)
    with seed_buf.map_to_host() as h_seed:
        for b in range(batch_size):
            h_seed[b] = UInt64(42)
    with temp_buf.map_to_host() as h_temp:
        for b in range(batch_size):
            h_temp[b] = Float32(1.0)

    var in_t = TileTensor(d_in, in_layout)
    var out_t = TileTensor(d_out, row_major(batch_size))
    var seed_t = (
        TileTensor(seed_buf, row_major(batch_size))
        .as_unsafe_any_origin()
        .as_immut()
    )
    var temp_t = (
        TileTensor(temp_buf, row_major(batch_size))
        .as_unsafe_any_origin()
        .as_immut()
    )

    ctx.enqueue_function[poison_lds_kernel[2048]](
        grid_dim=batch_size, block_dim=BLOCK
    )

    topk_topp_sampling_from_prob[DTYPE, IDX, BLOCK, from_logits=True](
        ctx,
        in_t,
        out_t,
        n,
        top_p_val=top_p,
        rng_seed=seed_t,
        temperature=temp_t,
    )
    ctx.synchronize()

    with d_out.map_to_host() as h_out:
        for b in range(batch_size):
            var v = Int(h_out[b])
            print("  ", name, "row", b, "kind", row_kind[b], "-> id", v)
            assert_true(
                v >= 0 and v < n,
                String("sampled index out of range in ")
                + name
                + " row "
                + String(b)
                + ": "
                + String(v),
            )

    _ = d_in^
    _ = d_out^
    _ = seed_buf^
    _ = temp_buf^


def check_degenerate_row_probs(
    ctx: DeviceContext,
    name: String,
    n: Int,
    row_kind: List[Int],
    top_k: Int,
) raises:
    """Same gate for the top-k-only sampler, which consumes probabilities.

    Here a degenerate row is an all-zero or all-`NaN` distribution: nothing
    exceeds the initial pivot, so the fallback slot is read unwritten.
    """
    var batch_size = len(row_kind)
    var in_shape = IndexList[2](batch_size, n)
    var in_layout = row_major(Coord(in_shape))

    var d_in = ctx.enqueue_create_buffer[DTYPE](in_shape.flattened_length())
    var d_out = ctx.enqueue_create_buffer[IDX](batch_size)

    with d_in.map_to_host() as h_in:
        var t = TileTensor(h_in, in_layout)
        for b in range(batch_size):
            var kind = row_kind[b]
            for i in range(n):
                if kind == ROW_ALL_ZERO:
                    t[b, i] = Scalar[DTYPE](0.0)
                elif kind == ROW_ALL_NAN:
                    t[b, i] = nan[DTYPE]()
                else:
                    t[b, i] = Scalar[DTYPE](1.0 / Float32(n))

    with d_out.map_to_host() as h_out:
        for b in range(batch_size):
            h_out[b] = Scalar[IDX](-7)

    var in_t = TileTensor(d_in, in_layout).as_immut()
    var out_t = TileTensor(d_out, row_major(batch_size))

    ctx.enqueue_function[poison_lds_kernel[2048]](
        grid_dim=batch_size, block_dim=BLOCK
    )

    topk_sampling_from_prob[DTYPE, IDX, BLOCK](
        ctx, in_t, out_t, top_k, rng_seed=42
    )
    ctx.synchronize()

    with d_out.map_to_host() as h_out:
        for b in range(batch_size):
            var v = Int(h_out[b])
            print("  ", name, "row", b, "kind", row_kind[b], "-> id", v)
            assert_true(
                v >= 0 and v < n,
                String("sampled index out of range in ")
                + name
                + " row "
                + String(b)
                + ": "
                + String(v),
            )

    _ = d_in^
    _ = d_out^


def main() raises:
    with DeviceContext() as ctx:
        var n = 8192
        check_degenerate_row(
            ctx, "well-formed", n, [ROW_WELL_FORMED, ROW_WELL_FORMED], 0.95
        )
        check_degenerate_row(
            ctx, "all -inf", n, [ROW_WELL_FORMED, ROW_ALL_NEG_INF], 0.95
        )
        check_degenerate_row(
            ctx, "single +inf", n, [ROW_WELL_FORMED, ROW_ONE_POS_INF], 0.95
        )
        check_degenerate_row(
            ctx, "all NaN", n, [ROW_WELL_FORMED, ROW_ALL_NAN], 0.95
        )

        # Top-k-only sampler, from probabilities.
        check_degenerate_row_probs(
            ctx, "probs all-zero", n, [ROW_WELL_FORMED, ROW_ALL_ZERO], 64
        )
        check_degenerate_row_probs(
            ctx, "probs all NaN", n, [ROW_WELL_FORMED, ROW_ALL_NAN], 64
        )
        print("PASS")
