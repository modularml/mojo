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
import std.math
from std.math import sqrt
from std.testing import assert_true
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
)

from layout import TileTensor, row_major
from attn_res.mix import attn_res_mix_gpu


def _cpu_ref[
    dtype: DType
](
    tokens: Int,
    C: Int,
    hidden: Int,
    v_h: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    proj_h: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    norm_h: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    eps: Float32,
    out_h: MutPointer[Float32, MutUntrackedOrigin],
):
    for t in range(tokens):
        var scores = List[Float64]()
        for c in range(C):
            var sum_sq = Float64(0)
            var dot = Float64(0)
            for h in range(hidden):
                var val = v_h[(t * C + c) * hidden + h].cast[.float64]()
                var sw = proj_h[h].cast[.float64]() * norm_h[h].cast[.float64]()
                sum_sq += val * val
                dot += val * sw
            var rms_scale = 1.0 / sqrt(sum_sq / Float64(hidden) + Float64(eps))
            scores.append(rms_scale * dot)
        var max_s = scores[0]
        for c in range(1, C):
            if scores[c] > max_s:
                max_s = scores[c]
        var denom = Float64(0)
        var probs = List[Float64]()
        for c in range(C):
            var e = std.math.exp(scores[c] - max_s)
            probs.append(e)
            denom += e
        for c in range(C):
            probs[c] = probs[c] / denom
        for h in range(hidden):
            var acc = Float64(0)
            for c in range(C):
                acc += probs[c] * v_h[(t * C + c) * hidden + h].cast[.float64]()
            out_h[t * hidden + h] = Float32(acc)


def _producer_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: ImmPointer[Float32, ImmutAnyOrigin],
    n: Int32,
    spin: Int32,
    sink: MutPointer[Float32, MutAnyOrigin],
):
    """Stand-in for the real graph's `concat`: a GPU grid that writes the
    exact buffer `attn_res_mix_gpu` reads next, immediately before it on the
    same stream. A DMA `enqueue_copy` predecessor would put host-side stream
    ordering in charge and never exercise the grid-to-grid handshake.

    The order -- stall, write, then `launch_dependent_grids()` -- is the
    producer's half of the PDL contract: publish only once the data the
    dependent reads is actually written. A producer that signals only by
    completing (the implicit trigger) never exercises the handshake, since
    nothing can be dispatched before the grid ends.

    Small and grid-strided on purpose, not one block per element: the driver
    only places a dependent grid early when SMs are free, so a producer sized
    to fill the machine keeps the consumer queued until the producer drains.
    """
    # Opaque dependent chain: seeded from a runtime load and retired to `sink`
    # so neither the loop nor the stall can be folded away.
    var acc = src[0]
    for _ in range(Int(spin)):
        acc = acc * Float32(1.0000001) + Float32(1e-9)

    var start = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    if start == 0:
        sink[0] = acc
    for i in range(start, Int(n), stride):
        dst[i] = src[i]

    # Publish: the dependent grid's `griddepcontrol.wait` unblocks here, once
    # every block of this grid has written its slice of `dst`.
    launch_dependent_grids()


def _assert_finite_contract[
    dtype: DType
](
    got: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    want: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    context: String,
) raises:
    """Asserts a finite reference never yields a non-finite kernel output.

    The relative-error checks at the call sites cannot do this themselves:
    `NaN > max_abs_err` is false, so a NaN output never updates the
    accumulator and an all-NaN result scores a rel_err of 0 -- measured, not
    assumed: with this assert disabled and the kernel's output store poisoned
    to NaN, all eight fixture cases pass. NaN is the one output this kernel
    is built to avoid -- the `rms_scale == 0` guard in `mix.mojo` exists
    because the reassociated form otherwise evaluates `0 * inf` -- so the
    harness has to be able to see it. Same contract as `numeric_check` in
    `test/gpu/fuzz/_fuzz.mojo`.
    """
    var n_bad = 0
    var first_bad = -1
    for i in range(n):
        if Bool(std.math.isfinite(want[i])) and not Bool(
            std.math.isfinite(got[i])
        ):
            n_bad += 1
            if first_bad < 0:
                first_bad = i
    assert_true(
        n_bad == 0,
        context
        + ": attn_res_mix_gpu returned "
        + String(n_bad)
        + " non-finite outputs where the reference is finite, first at index "
        + String(first_bad),
    )


def _check[
    dtype: DType, C: Int
](tokens: Int, hidden: Int, rel_tol: Float64, ctx: DeviceContext) raises:
    var v_h = alloc[Scalar[dtype]](tokens * C * hidden)
    var proj_h = alloc[Scalar[dtype]](hidden)
    var norm_h = alloc[Scalar[dtype]](hidden)
    var ref_out_h = alloc[Float32](tokens * hidden)

    # Quantize the fill through `dtype` so the fp64 reference consumes exactly
    # the values the device sees -- otherwise a low-precision run is charged
    # for input rounding the kernel never performed, and the tolerance stops
    # measuring the kernel.
    for i in range(tokens * C * hidden):
        v_h[i] = (
            std.math.sin(Float32(i + 1) * Float32(0.173)) * Float32(0.7)
        ).cast[dtype]()
    for i in range(hidden):
        proj_h[i] = std.math.cos(Float32(i + 3) * Float32(0.091)).cast[dtype]()
        norm_h[i] = (
            Float32(1.0)
            + std.math.sin(Float32(i + 2) * Float32(0.043)) * Float32(0.2)
        ).cast[dtype]()

    var eps = Float32(1e-6)
    _cpu_ref(tokens, C, hidden, v_h, proj_h, norm_h, eps, ref_out_h)

    var v_dev = ctx.enqueue_create_buffer[dtype](tokens * C * hidden)
    var proj_dev = ctx.enqueue_create_buffer[dtype](hidden)
    var norm_dev = ctx.enqueue_create_buffer[dtype](hidden)
    var out_dev = ctx.enqueue_create_buffer[dtype](tokens * hidden)

    with ctx.push_context():
        ctx.enqueue_copy(v_dev, v_h)
        ctx.enqueue_copy(proj_dev, proj_h)
        ctx.enqueue_copy(norm_dev, norm_h)

    var v_tt = TileTensor(v_dev, row_major(tokens, C, hidden))
    var proj_tt = TileTensor(proj_dev, row_major(1, hidden))
    var norm_tt = TileTensor(norm_dev, row_major(hidden))
    var out_tt = TileTensor(out_dev, row_major(tokens, hidden))

    comptime BLOCK = 256
    ctx.enqueue_function[
        attn_res_mix_gpu[
            dtype,
            out_tt.LayoutType,
            out_tt.Engine,
            v_tt.LayoutType,
            v_tt.Engine,
            proj_tt.LayoutType,
            proj_tt.Engine,
            norm_tt.LayoutType,
            norm_tt.Engine,
            C,
            BLOCK,
        ]
    ](
        out_tt,
        v_tt,
        proj_tt,
        norm_tt,
        eps,
        Int32(hidden),
        grid_dim=(tokens,),
        block_dim=(BLOCK,),
    )

    var out_h = alloc[Scalar[dtype]](tokens * hidden)
    with ctx.push_context():
        ctx.enqueue_copy(out_h, out_dev)
    ctx.synchronize()

    _assert_finite_contract(
        out_h, ref_out_h, tokens * hidden, "dtype=" + String(dtype)
    )

    var max_abs_err = Float64(0)
    var max_abs_ref = Float64(0)
    for i in range(tokens * hidden):
        var got = out_h[i].cast[.float64]()
        var want = ref_out_h[i].cast[.float64]()
        var err = abs(got - want)
        if err > max_abs_err:
            max_abs_err = err
        if abs(want) > max_abs_ref:
            max_abs_ref = abs(want)
    var rel_err = max_abs_err / (max_abs_ref + 1e-8)
    print(
        "dtype="
        + String(dtype)
        + " tokens="
        + String(tokens)
        + " C="
        + String(C)
        + " hidden="
        + String(hidden)
        + " max_abs_err="
        + String(max_abs_err)
        + " rel_err="
        + String(rel_err)
        + " tol="
        + String(rel_tol)
    )
    assert_true(
        rel_err < rel_tol,
        "attn_res_mix_gpu rel_err too high for " + String(dtype),
    )

    v_h.free()
    proj_h.free()
    norm_h.free()
    ref_out_h.free()
    out_h.free()


# fp32 holds the fused reduction to a few ulps of the fp64 recompute (measured
# 2.6e-7 worst case across the cases below). bf16 carries 8 mantissa bits, so
# the output store alone costs ~2^-9 relative; measured 2.8e-3, and this keeps
# ~3.5x headroom on that.
comptime F32_REL_TOL = 1e-4
comptime BF16_REL_TOL = 1e-2


def test_single_candidate_fixture_hidden(ctx: DeviceContext) raises:
    """C=1 (no earlier candidates, softmax degenerates to a no-op)."""
    _check[.float32, 1](4, 1024, F32_REL_TOL, ctx)


def test_two_candidates_fixture_hidden(ctx: DeviceContext) raises:
    _check[.float32, 2](8, 1024, F32_REL_TOL, ctx)


def test_three_candidates_k3_mini_hidden(ctx: DeviceContext) raises:
    """K3-mini / real K3's own shape: hidden=7168, C up to 3 there."""
    _check[.float32, 3](8, 7168, F32_REL_TOL, ctx)


def test_eight_candidates_k3_mini_hidden(ctx: DeviceContext) raises:
    """Full K3 (93 layers, attn_res_block_size=12) reaches C=8."""
    _check[.float32, 8](2, 7168, F32_REL_TOL, ctx)


def test_single_token_decode_shape(ctx: DeviceContext) raises:
    """Decode: exactly 1 token, the shape this kernel exists to speed up."""
    _check[.float32, 3](1, 7168, F32_REL_TOL, ctx)


def test_bfloat16_k3_mini_hidden(ctx: DeviceContext) raises:
    """The dtype a served K3 actually instantiates. `dtype` is generic on the
    graph op (`builtin_kernels/attn_res.mojo`), so an fp32-only test leaves
    the production specialization uncompiled -- and it is the one whose input
    casts and output downcast are lossy."""
    _check[.bfloat16, 3](8, 7168, BF16_REL_TOL, ctx)


def test_bfloat16_single_token_decode_shape(ctx: DeviceContext) raises:
    """The bf16 decode shape: one token, so one CTA carries the whole
    reduction."""
    _check[.bfloat16, 3](1, 7168, BF16_REL_TOL, ctx)


def test_bfloat16_eight_candidates(ctx: DeviceContext) raises:
    """The bf16 case at the widest candidate axis, where the softmax has the
    most room to amplify a score error."""
    _check[.bfloat16, 8](2, 7168, BF16_REL_TOL, ctx)


def test_overflow_matches_unfused_semantics(ctx: DeviceContext) raises:
    """Finite inputs whose fp32 sum of squares AND score dot both saturate.

    This is the one regime where the fused reassociation and the reference's
    elementwise `k = v * rsqrt(mean(v*v))` are not the same expression (see
    the module docstring on `Kernels/lib/attn_res/mix.mojo`). The reference
    normalizes first, so every element of `k` is `finite * 0 == 0` and the
    score is a finite 0; the reassociated `rms_scale * total_dot` is
    `0 * inf == NaN` once the dot saturates too. Both sums have to saturate
    for the two to differ, hence the large `proj_weight` as well as large
    candidates -- a large candidate alone leaves the dot finite and both
    orders already agree.

    With all scores 0 the softmax is uniform, so the expected output is the
    plain mean of the candidates. An unguarded kernel returns NaN here.
    """
    comptime tokens = 2
    comptime C = 3
    comptime hidden = 1024
    comptime BLOCK = 256
    # 1e20 squares to 1e40, past fp32's 3.4e38, so a single element saturates
    # the sum regardless of `hidden`.
    comptime BIG = Float32(1e20)

    var v_h = alloc[Float32](tokens * C * hidden)
    var proj_h = alloc[Float32](hidden)
    var norm_h = alloc[Float32](hidden)
    for t in range(tokens):
        for c in range(C):
            for h in range(hidden):
                v_h[(t * C + c) * hidden + h] = Float32(c + 1) * BIG
    for h in range(hidden):
        proj_h[h] = BIG
        norm_h[h] = Float32(1.0)

    var v_dev = ctx.enqueue_create_buffer[.float32](tokens * C * hidden)
    var proj_dev = ctx.enqueue_create_buffer[.float32](hidden)
    var norm_dev = ctx.enqueue_create_buffer[.float32](hidden)
    var out_dev = ctx.enqueue_create_buffer[.float32](tokens * hidden)
    with ctx.push_context():
        ctx.enqueue_copy(v_dev, v_h)
        ctx.enqueue_copy(proj_dev, proj_h)
        ctx.enqueue_copy(norm_dev, norm_h)

    var v_tt = TileTensor(v_dev, row_major(tokens, C, hidden))
    var proj_tt = TileTensor(proj_dev, row_major(1, hidden))
    var norm_tt = TileTensor(norm_dev, row_major(hidden))
    var out_tt = TileTensor(out_dev, row_major(tokens, hidden))

    ctx.enqueue_function[
        attn_res_mix_gpu[
            DType.float32,
            out_tt.LayoutType,
            out_tt.Engine,
            v_tt.LayoutType,
            v_tt.Engine,
            proj_tt.LayoutType,
            proj_tt.Engine,
            norm_tt.LayoutType,
            norm_tt.Engine,
            C,
            BLOCK,
        ]
    ](
        out_tt,
        v_tt,
        proj_tt,
        norm_tt,
        Float32(1e-6),
        Int32(hidden),
        grid_dim=(tokens,),
        block_dim=(BLOCK,),
    )

    var out_h = alloc[Float32](tokens * hidden)
    with ctx.push_context():
        ctx.enqueue_copy(out_h, out_dev)
    ctx.synchronize()

    # Uniform weights over candidates 1e20, 2e20, 3e20.
    var expected = Float64(2e20)
    var worst = Float64(0)
    for i in range(tokens * hidden):
        var got = out_h[i].cast[.float64]()
        assert_true(
            not Bool(std.math.isnan(out_h[i])),
            (
                "attn_res_mix_gpu returned NaN on saturating finite input --"
                " the reassociated score hit 0 * inf where the reference"
                " gives a finite 0"
            ),
        )
        var rel = abs(got - expected) / expected
        if rel > worst:
            worst = rel
    print("overflow_regime worst_rel_err=" + String(worst))
    assert_true(
        worst < 1e-6,
        "attn_res_mix_gpu overflow output is not the uniform candidate mean",
    )

    v_h.free()
    proj_h.free()
    norm_h.free()
    out_h.free()


def test_pdl_no_host_sync_before_producer(ctx: DeviceContext) raises:
    """The real graph launches `concat` and `attn_res_mix` back-to-back on
    one stream with no host sync between them -- PDL's whole point. Every
    other test above drains the stream (`ctx.synchronize()`) before this
    kernel ever runs, which can't exercise that adjacency at all. This runs
    the whole handshake instead: `candidates` is poisoned, a producer grid
    stalls and then publishes it with `launch_dependent_grids()`, and
    `attn_res_mix_gpu` is launched behind it with `PDLLevel.ON` -- the same
    attribute the graph op uses -- with no host sync in between.

    What this does NOT do is fail when the kernel's `griddepcontrol.wait` is
    deleted; that was measured, not assumed. Whether a dependent grid is
    placed early enough to read pre-publish data is the driver's choice, and
    on SM100 it declines to for this shape: a standalone probe of the same
    producer/consumer geometry catches stale reads only a few times in 200
    iterations, and never through this kernel in 20. Arming PDL does not
    change that, so treat the check below as covering the launch
    configuration and the producer's half of the contract (publishing after
    its writes, not before), not as a guard on the consumer's wait. The
    in-tree precedent is the same shape: `test_matmul_pdl_race.mojo` catches a
    producer that signals too early, not a consumer that forgets to wait.
    """
    comptime tokens = 8
    comptime C = 3
    comptime hidden = 7168
    # Dependent-FMA stall, ~4ms, so the publish is unambiguously later than
    # the poison and the producer's writes rather than racing them.
    comptime PRODUCER_SPIN = 2000000
    comptime ITERATIONS = 20

    var proj_h = alloc[Float32](hidden)
    var norm_h = alloc[Float32](hidden)
    var ref_out_h = alloc[Float32](tokens * hidden)
    var out_h = alloc[Float32](tokens * hidden)
    for i in range(hidden):
        proj_h[i] = Float32(std.math.cos(Float32(i + 3) * Float32(0.091)))
        norm_h[i] = Float32(
            Float32(1.0)
            + std.math.sin(Float32(i + 2) * Float32(0.043)) * Float32(0.2)
        )
    var eps = Float32(1e-6)

    var proj_dev = ctx.enqueue_create_buffer[.float32](hidden)
    var norm_dev = ctx.enqueue_create_buffer[.float32](hidden)
    var v_dev = ctx.enqueue_create_buffer[.float32](tokens * C * hidden)
    var out_dev = ctx.enqueue_create_buffer[.float32](tokens * hidden)
    var sink_dev = ctx.enqueue_create_buffer[.float32](1)
    with ctx.push_context():
        ctx.enqueue_copy(proj_dev, proj_h)
        ctx.enqueue_copy(norm_dev, norm_h)

    var proj_tt = TileTensor(proj_dev, row_major(1, hidden))
    var norm_tt = TileTensor(norm_dev, row_major(hidden))
    var v_tt = TileTensor(v_dev, row_major(tokens, C, hidden))
    var out_tt = TileTensor(out_dev, row_major(tokens, hidden))
    comptime BLOCK = 256
    comptime PROD_BLOCK = 256
    # Small on purpose: leaves the machine free for the dependent grid to be
    # placed while the producer is still stalling. See `_producer_kernel`.
    comptime PROD_GRID = 8
    var n = tokens * C * hidden

    for iteration in range(ITERATIONS):
        var v_h = alloc[Float32](n)
        # Vary the data per iteration so a stale read (previous iteration's
        # or uninitialized `v_dev`) doesn't accidentally match by construction.
        for i in range(n):
            v_h[i] = Float32(
                std.math.sin(Float32(i + 1 + iteration * 97) * Float32(0.173))
                * Float32(0.7)
            )
        var staging_dev = ctx.enqueue_create_buffer[.float32](n)
        with ctx.push_context():
            ctx.enqueue_copy(staging_dev, v_h)
        ctx.synchronize()  # only the H2D upload syncs; not the risk surface.

        _cpu_ref(tokens, C, hidden, v_h, proj_h, norm_h, eps, ref_out_h)

        # Poison `v_dev` so an early read is a wrong ANSWER, not just an
        # unlucky one: all-zero candidates mix to an all-zero output, which is
        # nowhere near the reference.
        ctx.enqueue_memset[.float32](v_dev, 0)

        # Producer grid (stands in for `concat`) writes `v_dev`, then
        # `attn_res_mix_gpu` reads it -- back-to-back, same stream, no
        # `ctx.synchronize()` between them, matching the real graph.
        ctx.enqueue_function[_producer_kernel](
            v_dev.unsafe_ptr(),
            staging_dev.unsafe_ptr(),
            Int32(n),
            Int32(PRODUCER_SPIN),
            sink_dev.unsafe_ptr(),
            grid_dim=(PROD_GRID,),
            block_dim=(PROD_BLOCK,),
        )
        ctx.enqueue_function[
            attn_res_mix_gpu[
                DType.float32,
                out_tt.LayoutType,
                out_tt.Engine,
                v_tt.LayoutType,
                v_tt.Engine,
                proj_tt.LayoutType,
                proj_tt.Engine,
                norm_tt.LayoutType,
                norm_tt.Engine,
                C,
                BLOCK,
            ]
        ](
            out_tt,
            v_tt,
            proj_tt,
            norm_tt,
            eps,
            Int32(hidden),
            grid_dim=(tokens,),
            block_dim=(BLOCK,),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        with ctx.push_context():
            ctx.enqueue_copy(out_h, out_dev)
        ctx.synchronize()

        _assert_finite_contract(
            out_h,
            ref_out_h,
            tokens * hidden,
            "pdl_no_host_sync iteration=" + String(iteration),
        )

        var max_abs_err = Float64(0)
        var max_abs_ref = Float64(0)
        for i in range(tokens * hidden):
            var err = abs(Float64(out_h[i]) - Float64(ref_out_h[i]))
            if err > max_abs_err:
                max_abs_err = err
            if abs(Float64(ref_out_h[i])) > max_abs_ref:
                max_abs_ref = abs(Float64(ref_out_h[i]))
        var rel_err = max_abs_err / (max_abs_ref + 1e-8)
        print(
            "pdl_no_host_sync iteration="
            + String(iteration)
            + " max_abs_err="
            + String(max_abs_err)
            + " rel_err="
            + String(rel_err)
        )
        assert_true(
            rel_err < 1e-4,
            (
                "attn_res_mix_gpu rel_err too high with no host sync before it"
                " -- PDL wait may not be blocking on the producer grid"
            ),
        )
        v_h.free()

    proj_h.free()
    norm_h.free()
    ref_out_h.free()
    out_h.free()


def main() raises:
    with DeviceContext() as ctx:
        test_single_candidate_fixture_hidden(ctx)
        test_two_candidates_fixture_hidden(ctx)
        test_three_candidates_k3_mini_hidden(ctx)
        test_eight_candidates_k3_mini_hidden(ctx)
        test_single_token_decode_shape(ctx)
        test_bfloat16_k3_mini_hidden(ctx)
        test_bfloat16_single_token_decode_shape(ctx)
        test_bfloat16_eight_candidates(ctx)
        test_overflow_matches_unfused_semantics(ctx)
        test_pdl_no_host_sync_before_producer(ctx)
