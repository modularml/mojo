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
"""Correctness tests for `nn.topk.gumbel_sampling_fused_gpu[from_probs=True]`.

The kernel draws one token per row proportionally to a row of unnormalized
probabilities, by Gumbel-max over `ln(p)` with in-kernel noise seeded per
row. Speculative decoding samples its rejection residual with it.

The noise is a fixed function of `(seed, vocab_position)`, so the tests are
deterministic: the frequency check draws once per distinct seed and compares
pooled frequencies against the distribution, and the structural checks pin
the contracts the sampler relies on -- zero mass never wins, row scaling
does not matter, and equal seeds draw with equal noise.
"""

from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
)
from layout import TileTensor, row_major
from std.math import sqrt
from std.sys.info import has_amd_gpu_accelerator
from std.testing import assert_equal, assert_true

from nn.topk import _gumbel_argmax_fused_kernel, gumbel_sampling_fused_gpu


def _draw(
    ctx: DeviceContext,
    rows: Int,
    d: Int,
    probs: List[Float64],
    seeds: List[UInt64],
) raises -> List[Int]:
    """Runs one launch and returns the drawn token per row."""
    var probs_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    var seeds_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    for row in range(rows):
        for col in range(d):
            probs_host[row * d + col] = Float32(probs[row * d + col])
        seeds_host[row] = seeds[row]

    var probs_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    var seeds_dev = ctx.enqueue_create_buffer[.uint64](rows)
    var out_dev = ctx.enqueue_create_buffer[.int64](rows)
    ctx.enqueue_copy(probs_dev, probs_host)
    ctx.enqueue_copy(seeds_dev, seeds_host)

    gumbel_sampling_fused_gpu[from_probs=True](
        ctx,
        TileTensor(probs_dev, row_major(rows, d)),
        TileTensor(out_dev, row_major(rows)),
        seed=TileTensor(seeds_dev, row_major(rows))
        .as_unsafe_any_origin()
        .as_immut(),
    )

    var out_host = ctx.enqueue_create_host_buffer[.int64](rows)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var out = List[Int]()
    for row in range(rows):
        out.append(Int(out_host[row]))

    _ = probs_dev^
    _ = seeds_dev^
    _ = out_dev^
    return out^


def test_frequencies_match_distribution(ctx: DeviceContext) raises:
    """Pooled draw frequencies over distinct seeds match the distribution.

    Every row carries the same 8-way distribution (deliberately unnormalized)
    and its own seed, so the rows are independent draws. The tolerance is
    five binomial standard deviations plus slack; with fixed seeds the test
    is deterministic, the statistics only size the bound.
    """
    comptime d = 8
    var rows = 16384
    var weights: List[Float64] = [
        0.30,
        0.20,
        0.15,
        0.12,
        0.10,
        0.08,
        0.04,
        0.01,
    ]

    var probs = List[Float64]()
    var seeds = List[UInt64]()
    for row in range(rows):
        for col in range(d):
            # Unnormalized on purpose: scaled by 3.
            probs.append(weights[col] * 3.0)
        seeds.append(UInt64(row))

    var out = _draw(ctx, rows, d, probs, seeds)

    var counts = List[Int]()
    for _ in range(d):
        counts.append(0)
    for row in range(rows):
        counts[out[row]] += 1

    for col in range(d):
        var p = weights[col]
        var freq = Float64(counts[col]) / Float64(rows)
        var tol = 5.0 * sqrt(p * (1.0 - p) / Float64(rows)) + 1e-3
        assert_true(
            abs(freq - p) < tol,
            String(t"token {col}: frequency {freq} != {p} (tol {tol})"),
        )


def test_zero_mass_never_wins(ctx: DeviceContext, d: Int) raises:
    """Tokens with zero probability are never drawn.

    Rows carry mass on exactly one token (a different one per row), so the
    draw is forced. This is the property that lets the rejection sampler use
    `max(p - q, 0)` rows directly: zeroed-out tokens cannot resurface. The
    odd `d` also exercises the kernel's ragged tail path.
    """
    var rows = 64

    var probs = List[Float64]()
    var seeds = List[UInt64]()
    for row in range(rows):
        for col in range(d):
            probs.append(0.5 if col == (row * 7919) % d else 0.0)
        seeds.append(UInt64(row * 31 + 7))

    var out = _draw(ctx, rows, d, probs, seeds)
    for row in range(rows):
        assert_equal(
            out[row],
            (row * 7919) % d,
            String(t"row {row}: drew a zero-mass token {out[row]}"),
        )


def test_equal_seeds_share_noise(ctx: DeviceContext) raises:
    """Rows with equal probabilities and equal seeds draw the same token.

    Speculative decoding passes one seed per request, repeated across its
    draft positions, so that a request's positions race against one shared
    noise row. Scaling a row must not change its draw either, since the
    input is unnormalized.
    """
    comptime d = 512
    var rows = 6
    var seeds: List[UInt64] = [42, 42, 43, 42, 1, 2]

    var probs = List[Float64]()
    for row in range(rows):
        for col in range(d):
            var h = (col * 2654435761) % 1_000_003
            var p = Float64(h) / 1_000_003.0 + 0.001
            # Rows 0-2 and 4-5 share one distribution; row 3 is it scaled.
            probs.append(p * 1000.0 if row == 3 else p)

    var out = _draw(ctx, rows, d, probs, seeds)
    assert_equal(out[0], out[1], "equal seed + equal probs must agree")
    assert_equal(out[0], out[3], "scaling a row must not change the draw")
    # Different seeds are only overwhelmingly likely to differ; assert the
    # specific fixed seeds here do, so a stuck RNG cannot pass silently.
    var pair_a_differs = out[0] != out[2]
    var pair_b_differs = out[4] != out[5]
    assert_true(
        pair_a_differs or pair_b_differs,
        "distinct seeds produced identical draws for every pair",
    )


def test_empty_batch(ctx: DeviceContext) raises:
    """Zero rows must be a no-op, not a zero-sized grid launch.

    The rejection sampler runs with no rows on every step that has no drafts
    to verify, so the empty batch is a normal input.
    """
    comptime d = 1024
    var probs_dev = ctx.enqueue_create_buffer[.float32](0)
    var out_dev = ctx.enqueue_create_buffer[.int64](0)
    gumbel_sampling_fused_gpu[from_probs=True](
        ctx,
        TileTensor(probs_dev, row_major(0, d)),
        TileTensor(out_dev, row_major(0)),
    )
    ctx.synchronize()
    _ = probs_dev^
    _ = out_dev^


def test_amd_split_matches_single(ctx: DeviceContext, rows: Int, d: Int) raises:
    comptime hw_info = ctx.default_device_info
    comptime block_size = hw_info.max_thread_block_size
    var blocks_per_row = Int32(hw_info.sm_count // rows)
    assert_true(blocks_per_row > 1, "test shape must select multiple blocks")

    var probs_host = ctx.enqueue_create_host_buffer[.float32](rows * d)
    var seeds_host = ctx.enqueue_create_host_buffer[.uint64](rows)
    for row in range(rows):
        seeds_host[row] = UInt64(0xC001D00D + row * 104729)
        for col in range(d):
            var key = (row * 8191 + col * 65537) % 1009
            var probability = Float32(key + 1) / 1009.0
            if key % 17 == 0:
                probability = 0.0
            elif key % 13 < 2:
                probability = 0.25
            probs_host[row * d + col] = probability

    var probs_dev = ctx.enqueue_create_buffer[.float32](rows * d)
    var seeds_dev = ctx.enqueue_create_buffer[.uint64](rows)
    var single_out_dev = ctx.enqueue_create_buffer[.int64](rows)
    var split_out_dev = ctx.enqueue_create_buffer[.int64](rows)
    var partials_dev = ctx.enqueue_create_buffer[.uint8](
        rows * Int(blocks_per_row) * 16
    )
    var counters_dev = ctx.enqueue_create_buffer[.int32](rows)
    ctx.enqueue_copy(probs_dev, probs_host)
    ctx.enqueue_copy(seeds_dev, seeds_host)
    ctx.enqueue_memset(counters_dev, Int32(0))

    var probs = (
        TileTensor(probs_dev, row_major(rows, d))
        .as_unsafe_any_origin()
        .as_immut()
    )
    var seeds = (
        TileTensor(seeds_dev, row_major(rows)).as_unsafe_any_origin().as_immut()
    )
    var single_out = TileTensor(single_out_dev, row_major(rows))
    var split_out = TileTensor(split_out_dev, row_major(rows))
    var no_temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    var no_partials: Optional[UnsafePointer[UInt8, MutAnyOrigin]] = None
    var no_counters: Optional[UnsafePointer[Int32, MutAnyOrigin]] = None
    var partials = Optional[UnsafePointer[UInt8, MutAnyOrigin]](
        partials_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    )
    var counters = Optional[UnsafePointer[Int32, MutAnyOrigin]](
        counters_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    )

    comptime single_kernel = _gumbel_argmax_fused_kernel[
        DType.float32,
        DType.int64,
        probs.LayoutType,
        single_out.LayoutType,
        from_probs=True,
        multi_block=False,
        InputEngine=probs.Engine,
        OutIdxEngine=single_out.Engine,
    ]
    comptime split_kernel = _gumbel_argmax_fused_kernel[
        DType.float32,
        DType.int64,
        probs.LayoutType,
        split_out.LayoutType,
        from_probs=True,
        multi_block=True,
        InputEngine=probs.Engine,
        OutIdxEngine=split_out.Engine,
    ]
    ctx.enqueue_function[single_kernel](
        probs,
        single_out,
        no_temperature,
        Optional(seeds.ptr),
        Int32(1),
        no_partials,
        no_counters,
        grid_dim=rows,
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    ctx.enqueue_function[split_kernel](
        probs,
        split_out,
        no_temperature,
        Optional(seeds.ptr),
        blocks_per_row,
        partials,
        counters,
        grid_dim=rows * Int(blocks_per_row),
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )

    with single_out_dev.map_to_host() as single_host, split_out_dev.map_to_host() as split_host:
        for row in range(rows):
            assert_equal(
                split_host[row],
                single_host[row],
                String(t"rows={rows}, d={d}, row={row}"),
            )

    _ = probs_dev^
    _ = seeds_dev^
    _ = single_out_dev^
    _ = split_out_dev^
    _ = partials_dev^
    _ = counters_dev^


def main() raises:
    with DeviceContext() as ctx:
        test_empty_batch(ctx)
        test_frequencies_match_distribution(ctx)
        test_zero_mass_never_wins(ctx, d=131)
        test_zero_mass_never_wins(ctx, d=1024)
        # Real Llama-3.1-8B vocabulary width.
        test_zero_mass_never_wins(ctx, d=128256)
        test_equal_seeds_share_noise(ctx)
        comptime if has_amd_gpu_accelerator():
            test_amd_split_matches_single(ctx, rows=32, d=200064)
            test_amd_split_matches_single(ctx, rows=32, d=200065)
            test_amd_split_matches_single(ctx, rows=96, d=200064)
            test_amd_split_matches_single(ctx, rows=96, d=200065)
